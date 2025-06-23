#!/usr/bin/env python3
"""
Enhanced training script for spam detection model

This script provides a command-line interface for training the spam detection model
using the enhanced ML pipeline with comprehensive preprocessing and advanced models.
"""

import argparse
import logging
import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from enhanced_trainer import EnhancedSpamTrainer
    ENHANCED_TRAINER_AVAILABLE = True
except ImportError as e:
    ENHANCED_TRAINER_AVAILABLE = False
    print(f"Warning: Enhanced trainer not available: {e}")

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train enhanced spam detection model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory containing training data')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory to save trained models')
    
    # Training arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--vocab-size', type=int, default=50000,
                       help='Vocabulary size for the model')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Training options
    parser.add_argument('--use-semi-supervised', action='store_true',
                       help='Enable semi-supervised learning')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with rule-based detection')
    
    # Utility arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting enhanced spam detection model training")
    logger.info(f"Arguments: {args}")
    
    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    
    try:
        if args.quick_test:
            # Quick rule-based test
            logger.info("Running quick rule-based test")
            
            test_messages = [
                ("Buy cheap viagra now!", True),
                ("Meeting tomorrow at 3pm", False),
                ("URGENT: You've won $1000000!", True),
                ("Thanks for the dinner invitation", False),
                ("Click here to claim your prize NOW!", True),
                ("Can you pick up milk on your way home?", False)
            ]
            
            # Simple rule-based patterns
            spam_patterns = [
                r'(?i)(viagra|cialis|pharmacy)',
                r'(?i)(win|won|winner).*(money|cash|prize)',
                r'(?i)(click|visit).*(link|website)',
                r'(?i)(free|cheap).*(offer|deal)',
                r'(?i)(urgent|act now|limited time)',
                r'(?i)(bitcoin|crypto|investment)',
                r'\$\d+',
                r'(?i)(loan|debt|credit)'
            ]
            
            import re
            correct_predictions = 0
            
            for message, expected_spam in test_messages:
                spam_score = sum(1 for pattern in spam_patterns if re.search(pattern, message))
                if len(re.findall(r'[A-Z]', message)) > len(message) * 0.5:
                    spam_score += 1
                
                is_spam = spam_score > 0
                if is_spam == expected_spam:
                    correct_predictions += 1
                
                logger.info(f"Message: '{message[:50]}...' -> Spam: {is_spam} (Expected: {expected_spam})")
            
            accuracy = correct_predictions / len(test_messages)
            logger.info(f"Quick test accuracy: {accuracy:.2f}")
            
            # Save simple model info
            model_info = {
                "version": "1.0.0-rule-based",
                "type": "rule-based",
                "accuracy": accuracy,
                "patterns_count": len(spam_patterns),
                "created_at": str(time.time())
            }
            
            with open(os.path.join(args.models_dir, "model_info.json"), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"\\n{'='*50}")
            print("QUICK TEST COMPLETED!")
            print(f"{'='*50}")
            print(f"Test Accuracy: {accuracy:.2f}")
            print(f"Model Type: Rule-based spam detection")
            print(f"Models saved to: {args.models_dir}")
            print(f"{'='*50}")
            
            return
        
        # Enhanced ML training
        if not ENHANCED_TRAINER_AVAILABLE:
            logger.error("Enhanced trainer is not available")
            print("ERROR: Enhanced trainer not available. Please ensure:")
            print("1. enhanced_trainer.py exists in src/ directory")
            print("2. Required dependencies are installed (scikit-learn, nltk, tqdm)")
            print("3. Run: pip install -r requirements.txt")
            sys.exit(1)
        
        logger.info("Using enhanced training pipeline with comprehensive ML models")
        
        # Initialize enhanced trainer
        enhanced_trainer = EnhancedSpamTrainer(
            data_dir=args.data_dir,
            models_dir=args.models_dir
        )
        
        # Train with enhanced pipeline
        logger.info("Starting enhanced training...")
        results = enhanced_trainer.train_models(
            test_size=args.test_size,
            vocab_size=args.vocab_size,
            use_semi_supervised=args.use_semi_supervised,
            n_folds=args.cv_folds
        )
        
        # Print results
        logger.info("Enhanced training completed successfully!")
        if isinstance(results, dict):
            logger.info(f"Best model: {results.get('best_model', 'Unknown')}")
            logger.info(f"Best accuracy: {results.get('best_accuracy', 0):.3f}")
            
            print(f"\\n{'='*50}")
            print("ENHANCED TRAINING COMPLETED!")
            print(f"{'='*50}")
            print(f"Best Model: {results.get('best_model', 'Unknown')}")
            print(f"Best Accuracy: {results.get('best_accuracy', 0):.3f}")
            print(f"Cross-validation folds: {args.cv_folds}")
            print(f"Semi-supervised learning: {'Enabled' if args.use_semi_supervised else 'Disabled'}")
            print(f"Models saved to: {args.models_dir}")
            print(f"{'='*50}")
        else:
            logger.info("Training completed with basic results")
            print(f"\\n{'='*50}")
            print("TRAINING COMPLETED!")
            print(f"{'='*50}")
            print(f"Models saved to: {args.models_dir}")
            print(f"{'='*50}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\\nERROR: Training failed - {e}")
        print("\\nTroubleshooting:")
        print("1. Check that data files exist in the data directory")
        print("2. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("3. Check disk space and permissions")
        print("4. Try running with --quick-test for basic functionality")
        sys.exit(1)

if __name__ == "__main__":
    main()
