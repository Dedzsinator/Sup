"""
Training script for spam detection model

This script provides a command-line interface for training the spam detection model
with various options and configurations.
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
except ImportError:
    ENHANCED_TRAINER_AVAILABLE = False
    
try:
    from trainer import SpamDetectionTrainer
    from config.settings import load_config
    ORIGINAL_TRAINER_AVAILABLE = True
except ImportError:
    ORIGINAL_TRAINER_AVAILABLE = False

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
    parser = argparse.ArgumentParser(description='Train spam detection model')
    
    parser.add_argument('--dataset', type=str, default='spam_dataset',
                       help='Dataset name to load from data directory')
    parser.add_argument('--vocab-size', type=int, default=50000,
                       help='Vocabulary size for the model')
    parser.add_argument('--user-vector-dim', type=int, default=512,
                       help='User vector dimension')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--hyperopt', action='store_true',
                       help='Perform hyperparameter optimization')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic training data')
    parser.add_argument('--synthetic-samples', type=int, default=10000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory path')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Models directory path')
    parser.add_argument('--config-dir', type=str, default='./config',
                       help='Config directory path')
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced training pipeline with advanced ML techniques')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode with simplified training')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting spam detection model training")
    logger.info(f"Arguments: {args}")
    
    try:
        if args.enhanced and ENHANCED_TRAINER_AVAILABLE:
            # Use enhanced training pipeline
            logger.info("Using enhanced training pipeline with advanced ML techniques")
            
            # Initialize enhanced trainer
            enhanced_trainer = EnhancedSpamTrainer(
                data_dir=args.data_dir,
                models_dir=args.models_dir,
                config_dir=args.config_dir
            )
            
            # Load and preprocess data
            data_stats = enhanced_trainer.load_and_preprocess_data(
                use_email_data=True,
                use_sms_data=True,
                use_text_files=True,
                save_processed=True
            )
            
            logger.info(f"Data loaded: {data_stats['total_messages']} messages, {data_stats['feature_count']} features")
            
            # Train models with cross-validation
            training_results = enhanced_trainer.train_models(
                models_to_train=['multinomial_nb', 'complement_nb', 'logistic', 'random_forest', 'ensemble'],
                use_cross_validation=True,
                use_hyperparameter_tuning=args.hyperopt,
                cv_folds=5
            )
            
            # Evaluate best model
            evaluation_results = enhanced_trainer.evaluate_model(
                model_name=training_results['best_model'],
                test_size=args.test_size
            )
            
            # Save models
            enhanced_trainer.save_models(training_results['best_model'])
            
            # Generate report
            report = enhanced_trainer.generate_comprehensive_report()
            report_path = Path(args.config_dir) / "enhanced_training_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Print results
            logger.info("Enhanced training completed successfully!")
            logger.info(f"Best model: {training_results['best_model']}")
            logger.info(f"Test accuracy: {evaluation_results['metrics']['accuracy']:.3f}")
            logger.info(f"Test F1 score: {evaluation_results['metrics']['f1']:.3f}")
            
            print("\n" + "="*60)
            print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Best Model: {training_results['best_model']}")
            print(f"Test Accuracy: {evaluation_results['metrics']['accuracy']:.3f}")
            print(f"Test Precision: {evaluation_results['metrics']['precision']:.3f}")
            print(f"Test Recall: {evaluation_results['metrics']['recall']:.3f}")
            print(f"Test F1 Score: {evaluation_results['metrics']['f1']:.3f}")
            if 'roc_auc' in evaluation_results['metrics']:
                print(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.3f}")
            print(f"Data: {data_stats['total_messages']} messages, {data_stats['feature_count']} features")
            print(f"Report saved to: {report_path}")
            print("="*60)
            
            return
            
        elif args.quick:
            # Quick test mode
            if ENHANCED_TRAINER_AVAILABLE:
                logger.info("Quick test mode with enhanced trainer")
                enhanced_trainer = EnhancedSpamTrainer(
                    data_dir=args.data_dir,
                    models_dir=args.models_dir,
                    config_dir=args.config_dir
                )
                
                quick_results = enhanced_trainer.quick_test()
                
                print("\n" + "="*50)
                print("ENHANCED QUICK TEST COMPLETED!")
                print("="*50)
                print(f"Test Accuracy: {quick_results['accuracy']:.3f}")
                print(f"Sample Predictions:")
                for i, (msg, true_label, pred, prob) in enumerate(zip(
                    quick_results['test_messages'][:3], 
                    quick_results['true_labels'][:3],
                    quick_results['predictions'][:3],
                    quick_results['probabilities'][:3]
                )):
                    label_name = 'spam' if pred == 1 else 'ham'
                    true_name = 'spam' if true_label == 1 else 'ham'
                    print(f"  {i+1}. '{msg[:40]}...' -> {label_name} ({prob:.3f}) [True: {true_name}]")
                print("Enhanced pipeline ready for production!")
                print("="*50)
                
                return
            else:
                # Fallback to original quick test
                logger.info("Quick test mode - testing basic spam detection")
                
                # Create directories if they don't exist
                os.makedirs(args.data_dir, exist_ok=True)
                os.makedirs(args.models_dir, exist_ok=True) 
                os.makedirs(args.config_dir, exist_ok=True)
                
                # Test basic spam detection patterns
                test_messages = [
                    ("Hello, how are you?", False),
                    ("BUY VIAGRA NOW!!! CHEAP PRICES!!!", True),
                    ("Win $1000000 now! Click this link!", True),
                    ("Thanks for the meeting today", False),
                    ("URGENT: Nigerian prince needs help", True)
                ]
                
                logger.info("Testing spam detection patterns...")
                correct_predictions = 0
                
                for message, expected_spam in test_messages:
                    # Simple spam detection using patterns from test_server.py
                    import re
                    spam_patterns = [
                        r'(?i)(viagra|cialis|pharmacy)',
                        r'(?i)(win|won|winner).*(money|cash|prize)',
                        r'(?i)(click|visit).*(link|website)',
                        r'(?i)(free|cheap).*(offer|deal)',
                        r'(?i)(urgent|act now|limited time)',
                        r'(?i)(bitcoin|crypto|investment)',
                        r'\$\d+',
                        r'(?i)(loan|debt|credit)',
                        r'(?i)(weight loss|diet pill)',
                        r'(?i)(nigerian prince|inheritance)'
                    ]
                    
                    spam_score = sum(1 for pattern in spam_patterns if re.search(pattern, message))
                    if len(re.findall(r'[A-Z]', message)) > len(message) * 0.5:
                        spam_score += 1
                    if len(re.findall(r'[!?]{2,}', message)) > 0:
                        spam_score += 0.5
                    
                    is_spam = spam_score > 1
                    if is_spam == expected_spam:
                        correct_predictions += 1
                    
                    logger.info(f"Message: '{message[:50]}...' -> Spam: {is_spam} (Expected: {expected_spam})")
                
                accuracy = correct_predictions / len(test_messages)
                logger.info(f"Quick test accuracy: {accuracy:.2f}")
                
                # Create a simple model info file
                model_info = {
                    "version": "1.0.0-test",
                    "type": "rule-based",
                    "accuracy": accuracy,
                    "patterns_count": len(spam_patterns),
                    "created_at": str(time.time())
                }
                
                import json
                with open(os.path.join(args.models_dir, "model_info.json"), 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                print("\n" + "="*50)
                print("QUICK TEST COMPLETED SUCCESSFULLY!")
                print("="*50)
                print(f"Test Accuracy: {accuracy:.2f}")
                print(f"Model Type: Rule-based spam detection")
                print(f"Ready for testing with test_server.py")
                print("="*50)
                
                return
        
        # Original training pipeline
        if not ORIGINAL_TRAINER_AVAILABLE:
            logger.error("Neither enhanced nor original trainer is available")
            print("ERROR: Training modules not available. Please install required dependencies.")
            sys.exit(1)

        # Initialize trainer
        trainer = SpamDetectionTrainer(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            config_dir=args.config_dir
        )
        
        # Load or generate training data
        if args.generate_synthetic:
            logger.info(f"Generating {args.synthetic_samples} synthetic training samples")
            messages, labels, user_ids, timestamps = trainer._generate_synthetic_data(
                n_samples=args.synthetic_samples
            )
        else:
            logger.info(f"Loading training data from {args.dataset}")
            messages, labels, user_ids, timestamps = trainer.load_training_data(args.dataset)
        
        logger.info(f"Loaded {len(messages)} training samples")
        
        # Perform hyperparameter optimization if requested
        if args.hyperopt:
            logger.info("Starting hyperparameter optimization")
            hyperopt_results = trainer.hyperparameter_optimization(
                messages, labels, user_ids, timestamps
            )
            logger.info(f"Best hyperparameters: {hyperopt_results['best_params']}")
            logger.info(f"Best score: {hyperopt_results['best_score']:.3f}")
            
            # Use best hyperparameters for training
            best_params = hyperopt_results['best_params']
            vocab_size = best_params['vocab_size']
            user_vector_dim = best_params['user_vector_dim']
        else:
            vocab_size = args.vocab_size
            user_vector_dim = args.user_vector_dim
        
        # Train the model
        logger.info("Starting model training")
        results = trainer.train_model(
            messages=messages,
            labels=labels,
            user_ids=user_ids,
            timestamps=timestamps,
            test_size=args.test_size,
            vocab_size=vocab_size,
            user_vector_dim=user_vector_dim
        )
        
        # Print results
        logger.info("Training completed successfully!")
        logger.info(f"Training time: {results['training_time_seconds']:.2f} seconds")
        logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.3f}")
        logger.info(f"Test precision: {results['test_metrics']['precision']:.3f}")
        logger.info(f"Test recall: {results['test_metrics']['recall']:.3f}")
        logger.info(f"Test F1 score: {results['test_metrics']['f1']:.3f}")
        logger.info(f"Test ROC AUC: {results['test_metrics']['roc_auc']:.3f}")
        
        # Generate and save training report
        report = trainer.generate_training_report()
        report_path = Path(args.config_dir) / "training_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Training report saved to {report_path}")
        
        # Print success message
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Final F1 Score: {results['test_metrics']['f1']:.3f}")
        print(f"Final Accuracy: {results['test_metrics']['accuracy']:.3f}")
        print(f"Model saved and ready for deployment")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\nERROR: Training failed - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
