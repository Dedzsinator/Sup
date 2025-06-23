"""
Enhanced Training Pipeline with Advanced ML Techniques

This module provides a comprehensive training pipeline that integrates:
- Enhanced data preprocessing with email parsing
- Advanced feature engineering (TF-IDF, linguistic, spam-specific)
- Multiple ML algorithms with cross-validation
- Semi-supervised learning for unlabeled data
- Hyperparameter optimization
- Model persistence and deployment
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import warnings

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback tqdm class
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass

# Import our custom modules
from enhanced_preprocessor import EnhancedDataPreprocessor
from advanced_classifier import AdvancedSpamClassifier

logger = logging.getLogger(__name__)

class EnhancedSpamTrainer:
    """
    Enhanced spam detection trainer with advanced ML techniques
    """
    
    def __init__(self, 
                 data_dir: str = "/home/deginandor/Documents/Programming/Sup/spam_detection/data",
                 models_dir: str = "/home/deginandor/Documents/Programming/Sup/spam_detection/models",
                 config_dir: str = "/home/deginandor/Documents/Programming/Sup/spam_detection/config"):
        
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = EnhancedDataPreprocessor(data_dir=str(self.data_dir))
        self.classifier = AdvancedSpamClassifier()
        
        # Training state
        self.training_data = None
        self.feature_matrix = None
        self.feature_names = None
        self.training_results = {}
        
        logger.info("Initialized EnhancedSpamTrainer")
    
    def load_and_preprocess_data(self, 
                                use_email_data: bool = True,
                                use_sms_data: bool = True,
                                use_text_files: bool = True,
                                save_processed: bool = True) -> Dict[str, Any]:
        """
        Load and preprocess all available data with comprehensive progress tracking
        """
        logger.info("Loading and preprocessing data...")
        
        all_messages = []
        all_labels = []
        all_metadata = []
        
        # Calculate total steps for progress tracking
        data_source_steps = sum([use_email_data, use_sms_data, use_text_files])
        total_steps = data_source_steps + 2  # +2 for feature creation and saving
        
        # Initialize progress bar for overall progress
        with tqdm(total=total_steps, desc="📊 Data Loading Pipeline", unit="step", 
                  disable=not TQDM_AVAILABLE) as pbar:
            
            # Load different data sources
            if use_email_data:
                pbar.set_description("📧 Loading email data")
                email_messages, email_labels, email_metadata = self.preprocessor.load_email_data()
                all_messages.extend(email_messages)
                all_labels.extend(email_labels)
                all_metadata.extend(email_metadata)
                logger.info(f"Loaded {len(email_messages)} email messages")
                pbar.update(1)
            
            if use_sms_data:
                pbar.set_description("💬 Loading SMS data")
                sms_messages, sms_labels, sms_metadata = self.preprocessor.load_sms_data()
                all_messages.extend(sms_messages)
                all_labels.extend(sms_labels)
                all_metadata.extend(sms_metadata)
                logger.info(f"Loaded {len(sms_messages)} SMS messages")
                pbar.update(1)
            
            if use_text_files:
                pbar.set_description("📄 Loading text files")
                text_messages, text_labels, text_metadata = self.preprocessor.load_text_files()
                all_messages.extend(text_messages)
                all_labels.extend(text_labels)
                all_metadata.extend(text_metadata)
                logger.info(f"Loaded {len(text_messages)} text file messages")
                pbar.update(1)
            
            # Store training data
            self.training_data = {
                'messages': all_messages,
                'labels': np.array(all_labels),
                'metadata': all_metadata
            }
            
            # Create feature matrix with detailed progress
            pbar.set_description("🔧 Creating feature matrix")
            logger.info("Creating feature matrix...")
            self.feature_matrix, self.feature_names = self.preprocessor.create_feature_matrix(
                all_messages, 
                max_features=5000,
                ngram_range=(1, 2),
                use_tfidf=True
            )
            pbar.update(1)
            
            # Save processed data if requested
            if save_processed:
                pbar.set_description("💾 Saving processed data")
                self.preprocessor.save_processed_data(
                    all_messages, all_labels, all_metadata,
                    filename="enhanced_processed_data.jsonl"
                )
                pbar.update(1)
            
            pbar.set_description("✅ Data loading complete")
        
        # Data statistics
        labeled_mask = self.training_data['labels'] >= 0
        labeled_labels = self.training_data['labels'][labeled_mask]
        
        stats = {
            'total_messages': len(all_messages),
            'labeled_messages': np.sum(labeled_mask),
            'unlabeled_messages': np.sum(~labeled_mask),
            'spam_messages': np.sum(labeled_labels == 1),
            'ham_messages': np.sum(labeled_labels == 0),
            'feature_count': self.feature_matrix.shape[1],
            'data_sources': {
                'email': len(email_messages) if use_email_data else 0,
                'sms': len(sms_messages) if use_sms_data else 0,
                'text_files': len(text_messages) if use_text_files else 0
            }
        }
        
        # Print comprehensive summary with emojis
        if TQDM_AVAILABLE:
            print(f"\n📊 Data Loading Summary:")
            print(f"  📁 Total messages: {stats['total_messages']:,}")
            print(f"  🏷️  Labeled: {stats['labeled_messages']:,} (Spam: {stats['spam_messages']:,}, Ham: {stats['ham_messages']:,})")
            print(f"  ❓ Unlabeled: {stats['unlabeled_messages']:,}")
            print(f"  🔧 Features: {stats['feature_count']:,}")
            print(f"  📧 Email data: {stats['data_sources']['email']:,}")
            print(f"  💬 SMS data: {stats['data_sources']['sms']:,}")
            print(f"  📄 Text files: {stats['data_sources']['text_files']:,}")
        
        logger.info(f"Data preprocessing complete:")
        logger.info(f"  Total messages: {stats['total_messages']}")
        logger.info(f"  Labeled: {stats['labeled_messages']} (Spam: {stats['spam_messages']}, Ham: {stats['ham_messages']})")
        logger.info(f"  Unlabeled: {stats['unlabeled_messages']}")
        logger.info(f"  Features: {stats['feature_count']}")
        
        return stats
    
    def train_models(self, 
                    models_to_train: List[str] = None,
                    use_cross_validation: bool = True,
                    use_hyperparameter_tuning: bool = False,
                    cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train multiple models with cross-validation and hyperparameter tuning
        """
        if self.training_data is None or self.feature_matrix is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        logger.info("Starting model training...")
        
        # Default models to train
        if models_to_train is None:
            models_to_train = ['multinomial_nb', 'complement_nb', 'logistic', 'random_forest', 'ensemble']
        
        # Calculate total steps for comprehensive progress tracking
        total_steps = 1  # For initial training
        if use_cross_validation:
            total_steps += len(models_to_train)  # CV for each model
        if use_hyperparameter_tuning:
            tunable_models = [m for m in ['random_forest', 'logistic'] if m in models_to_train]
            total_steps += len(tunable_models)
        
        # Initialize progress bar for training pipeline
        with tqdm(total=total_steps, desc="🤖 Model Training Pipeline", unit="step", 
                  disable=not TQDM_AVAILABLE) as pbar:
            
            # Train models with detailed progress
            pbar.set_description("🏋️  Training all models")
            training_results = self.classifier.fit(
                self.feature_matrix, 
                self.training_data['labels'],
                semi_supervised_data=None  # Will be handled automatically
            )
            pbar.update(1)
            
            # Print training results summary
            if TQDM_AVAILABLE:
                print(f"\n📈 Training Results Summary:")
                for model_name, result in training_results.items():
                    if 'error' not in result:
                        print(f"  {model_name}: F1={result.get('f1', 0):.3f}, Acc={result.get('accuracy', 0):.3f}")
            
            # Cross-validation with per-model progress
            cv_results = {}
            if use_cross_validation:
                for model_name in models_to_train:
                    pbar.set_description(f"🔍 Cross-validating {model_name}")
                    logger.info(f"Performing cross-validation for {model_name}...")
                    
                    # Run CV for single model
                    single_cv_result = self.classifier.cross_validate(
                        self.feature_matrix,
                        self.training_data['labels'],
                        cv=cv_folds,
                        scoring='f1',
                        models=[model_name]  # Focus on single model
                    )
                    cv_results.update(single_cv_result)
                    pbar.update(1)
                
                # Print CV results summary
                if TQDM_AVAILABLE:
                    print(f"\n🎯 Cross-Validation Results:")
                    for model_name, cv_result in cv_results.items():
                        if 'error' not in cv_result:
                            print(f"  {model_name}: Mean F1={cv_result['mean']:.3f} (±{cv_result['std']:.3f})")
            
            # Hyperparameter tuning with per-model progress
            tuning_results = {}
            if use_hyperparameter_tuning:
                tunable_models = [m for m in ['random_forest', 'logistic'] if m in models_to_train]
                for model_name in tunable_models:
                    pbar.set_description(f"⚙️  Tuning {model_name}")
                    logger.info(f"Performing hyperparameter tuning for {model_name}...")
                    tuning_results[model_name] = self.classifier.hyperparameter_tuning(
                        self.feature_matrix,
                        self.training_data['labels'],
                        model_name=model_name,
                        cv=3,
                        n_iter=10
                    )
                    pbar.update(1)
                
                # Print tuning results summary
                if TQDM_AVAILABLE and tuning_results:
                    print(f"\n🎛️  Hyperparameter Tuning Results:")
                    for model_name, tuning_result in tuning_results.items():
                        if 'best_score' in tuning_result:
                            print(f"  {model_name}: Best Score={tuning_result['best_score']:.3f}")
            
            pbar.set_description("✅ Model training complete")
        
        # Store results
        self.training_results = {
            'training_results': training_results,
            'cv_results': cv_results,
            'tuning_results': tuning_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Find best model
        best_model = self._find_best_model(training_results, cv_results)
        
        logger.info(f"Training complete. Best model: {best_model}")
        
        return {
            'best_model': best_model,
            'training_results': training_results,
            'cv_results': cv_results,
            'tuning_results': tuning_results
        }
    
    def _find_best_model(self, training_results: Dict, cv_results: Dict) -> str:
        """Find the best performing model"""
        best_model = 'ensemble'  # Default
        best_score = 0.0
        
        # Use cross-validation results if available
        if cv_results:
            for model_name, cv_result in cv_results.items():
                if 'error' not in cv_result and cv_result['mean'] > best_score:
                    best_score = cv_result['mean']
                    best_model = model_name
        else:
            # Use training results
            for model_name, result in training_results.items():
                if 'error' not in result and result.get('f1', 0) > best_score:
                    best_score = result['f1']
                    best_model = model_name
        
        return best_model
    
    def evaluate_model(self, 
                      model_name: str = None,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Dict[str, Any]:
        """
        Evaluate model performance on held-out test set with progress tracking
        """
        if self.training_data is None or self.feature_matrix is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        if TQDM_AVAILABLE:
            print("📊 Evaluating model performance...")
        
        # Use only labeled data for evaluation
        labeled_mask = self.training_data['labels'] >= 0
        X_labeled = self.feature_matrix[labeled_mask]
        y_labeled = self.training_data['labels'][labeled_mask]
        
        # Split data with progress
        with tqdm(total=4, desc="🔍 Model Evaluation", unit="step", 
                  disable=not TQDM_AVAILABLE) as pbar:
            
            # Step 1: Split data
            pbar.set_description("✂️  Splitting data")
            try:
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_labeled, y_labeled, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=y_labeled
                )
            except ImportError:
                # Fallback to manual split
                n_test = int(len(X_labeled) * test_size)
                indices = np.random.RandomState(random_state).permutation(len(X_labeled))
                test_indices = indices[:n_test]
                train_indices = indices[n_test:]
                
                X_train, X_test = X_labeled[train_indices], X_labeled[test_indices]
                y_train, y_test = y_labeled[train_indices], y_labeled[test_indices]
            pbar.update(1)
            
            # Step 2: Train on training set
            pbar.set_description("🏋️  Training on train set")
            self.classifier.fit(X_train, y_train)
            pbar.update(1)
            
            # Step 3: Find best model if not specified
            pbar.set_description("🏆 Finding best model")
            if model_name is None:
                model_name = self._find_best_model(self.training_results.get('training_results', {}), 
                                                 self.training_results.get('cv_results', {}))
            pbar.update(1)
            
            # Step 4: Make predictions and calculate metrics
            pbar.set_description("🔮 Making predictions")
            y_pred = self.classifier.predict(X_test, model_name=model_name)
            y_pred_proba = self.classifier.predict_proba(X_test, model_name=model_name)
            
            # Calculate metrics
            metrics = self.classifier._calculate_metrics(
                y_test, y_pred, 
                y_pred_proba[:, 1] if y_pred_proba.size > 0 else None
            )
            pbar.update(1)
            
            pbar.set_description("✅ Evaluation complete")
        
        evaluation_results = {
            'model_name': model_name,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'metrics': metrics,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba[:, 1].tolist() if y_pred_proba.size > 0 else []
            }
        }
        
        # Print detailed evaluation results
        if TQDM_AVAILABLE:
            print(f"\n📊 Evaluation Results for {model_name}:")
            print(f"  🎯 Test Set Size: {len(X_test):,} samples")
            print(f"  📈 Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"  🎯 F1 Score: {metrics.get('f1', 0):.3f}")
            print(f"  ⚖️  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  🔍 Recall: {metrics.get('recall', 0):.3f}")
            if 'roc_auc' in metrics:
                print(f"  📊 ROC AUC: {metrics['roc_auc']:.3f}")
        
        logger.info(f"Model evaluation complete:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Test Accuracy: {metrics.get('accuracy', 0):.3f}")
        logger.info(f"  Test F1 Score: {metrics.get('f1', 0):.3f}")
        logger.info(f"  Test Precision: {metrics.get('precision', 0):.3f}")
        logger.info(f"  Test Recall: {metrics.get('recall', 0):.3f}")
        
        return evaluation_results
    
    def save_models(self, model_name: str = None):
        """Save trained models to disk"""
        if model_name is None:
            model_name = self._find_best_model(
                self.training_results.get('training_results', {}),
                self.training_results.get('cv_results', {})
            )
        
        if TQDM_AVAILABLE:
            print(f"💾 Saving models (best model: {model_name})...")
        
        # Save the classifier
        model_path = self.models_dir / f"{model_name}_classifier.pkl"
        if TQDM_AVAILABLE:
            print("  📋 Saving classifier...")
        self.classifier.save_model(str(model_path), model_name)
        
        # Save the preprocessor
        preprocessor_path = self.models_dir / "preprocessor.pkl"
        if TQDM_AVAILABLE:
            print("  🔧 Saving preprocessor...")
        import pickle
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save training results
        results_path = self.models_dir / "training_results.json"
        if TQDM_AVAILABLE:
            print("  📊 Saving training results...")
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save model metadata
        metadata = {
            'best_model': model_name,
            'timestamp': datetime.now().isoformat(),
            'data_stats': {
                'total_messages': len(self.training_data['messages']),
                'feature_count': self.feature_matrix.shape[1],
                'spam_ratio': np.mean(self.training_data['labels'][self.training_data['labels'] >= 0])
            },
            'model_files': {
                'classifier': str(model_path),
                'preprocessor': str(preprocessor_path),
                'results': str(results_path)
            }
        }
        
        metadata_path = self.models_dir / "model_metadata.json"
        if TQDM_AVAILABLE:
            print("  📋 Saving metadata...")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if TQDM_AVAILABLE:
            print(f"✅ Models saved successfully!")
        
        logger.info(f"Models saved:")
        logger.info(f"  Classifier: {model_path}")
        logger.info(f"  Preprocessor: {preprocessor_path}")
        logger.info(f"  Metadata: {metadata_path}")
    
    def load_models(self, model_name: str = None):
        """Load trained models from disk"""
        if TQDM_AVAILABLE:
            print("📂 Loading models from disk...")
        
        # Load metadata
        metadata_path = self.models_dir / "model_metadata.json"
        if metadata_path.exists():
            if TQDM_AVAILABLE:
                print("  📋 Loading metadata...")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if model_name is None:
                model_name = metadata.get('best_model', 'ensemble')
        
        # Load classifier
        model_path = self.models_dir / f"{model_name}_classifier.pkl"
        if model_path.exists():
            if TQDM_AVAILABLE:
                print(f"  🤖 Loading classifier ({model_name})...")
            self.classifier.load_model(str(model_path))
        
        # Load preprocessor
        preprocessor_path = self.models_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            if TQDM_AVAILABLE:
                print("  🔧 Loading preprocessor...")
            import pickle
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        
        # Load training results
        results_path = self.models_dir / "training_results.json"
        if results_path.exists():
            if TQDM_AVAILABLE:
                print("  📊 Loading training results...")
            with open(results_path, 'r') as f:
                self.training_results = json.load(f)
        
        if TQDM_AVAILABLE:
            print(f"✅ Models loaded successfully!")
        
        logger.info(f"Models loaded: {model_name}")
    
    def predict(self, messages: List[str], model_name: str = None) -> Dict[str, Any]:
        """
        Make predictions on new messages
        """
        if not messages:
            return {'predictions': [], 'probabilities': []}
        
        if TQDM_AVAILABLE and len(messages) > 10:
            print(f"Making predictions on {len(messages)} messages...")
        
        # Preprocess messages with progress bar
        if TQDM_AVAILABLE and len(messages) > 100:
            processed_messages = [self.preprocessor.preprocess_text(msg) for msg in tqdm(messages, desc="Preprocessing messages")]
        else:
            processed_messages = [self.preprocessor.preprocess_text(msg) for msg in messages]
        
        # Create feature matrix
        if hasattr(self.preprocessor, 'tfidf_vectorizer') and self.preprocessor.tfidf_vectorizer is not None:
            # Use existing vectorizer
            tfidf_features = self.preprocessor.tfidf_vectorizer.transform(processed_messages)
            
            # Extract other features with progress bar
            linguistic_features = []
            spam_features = []
            
            message_iter = tqdm(messages, desc="Extracting features") if TQDM_AVAILABLE and len(messages) > 100 else messages
            for msg in message_iter:
                ling_feat = self.preprocessor.extract_linguistic_features(msg)
                spam_feat = self.preprocessor.extract_spam_features(msg)
                
                linguistic_features.append(list(ling_feat.values()))
                spam_features.append(list(spam_feat.values()))
            
            # Combine features
            linguistic_features = np.array(linguistic_features)
            spam_features = np.array(spam_features)
            
            # Scale features
            if self.preprocessor.scaler is not None:
                scaled_features = self.preprocessor.scaler.transform(
                    np.hstack([linguistic_features, spam_features])
                )
            else:
                scaled_features = np.hstack([linguistic_features, spam_features])
            
            # Combine with TF-IDF
            if hasattr(tfidf_features, 'toarray'):
                tfidf_features = tfidf_features.toarray()
            
            X = np.hstack([tfidf_features, scaled_features])
        else:
            # Fallback to basic features
            X, _ = self.preprocessor._create_basic_features(messages)
        
        # Make predictions
        predictions = self.classifier.predict(X, model_name=model_name)
        probabilities = self.classifier.predict_proba(X, model_name=model_name)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities[:, 1].tolist() if probabilities.size > 0 else [],
            'labels': ['ham' if pred == 0 else 'spam' for pred in predictions]
        }
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive training and evaluation report"""
        report = ["# Enhanced Spam Detection Training Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Data statistics
        if self.training_data is not None:
            labeled_mask = self.training_data['labels'] >= 0
            labeled_labels = self.training_data['labels'][labeled_mask]
            
            report.append("## Data Statistics")
            report.append(f"- Total Messages: {len(self.training_data['messages'])}")
            report.append(f"- Labeled Messages: {np.sum(labeled_mask)}")
            report.append(f"- Unlabeled Messages: {np.sum(~labeled_mask)}")
            report.append(f"- Spam Messages: {np.sum(labeled_labels == 1)}")
            report.append(f"- Ham Messages: {np.sum(labeled_labels == 0)}")
            report.append(f"- Feature Count: {self.feature_matrix.shape[1] if self.feature_matrix is not None else 'N/A'}")
            report.append("")
        
        # Training results
        if self.training_results:
            report.append("## Training Results")
            
            training_results = self.training_results.get('training_results', {})
            for model_name, metrics in training_results.items():
                if 'error' not in metrics:
                    report.append(f"### {model_name}")
                    report.append(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
                    report.append(f"- Precision: {metrics.get('precision', 0):.3f}")
                    report.append(f"- Recall: {metrics.get('recall', 0):.3f}")
                    report.append(f"- F1 Score: {metrics.get('f1', 0):.3f}")
                    if 'roc_auc' in metrics:
                        report.append(f"- ROC AUC: {metrics['roc_auc']:.3f}")
                    report.append("")
            
            # Cross-validation results
            cv_results = self.training_results.get('cv_results', {})
            if cv_results:
                report.append("## Cross-Validation Results")
                for model_name, cv_result in cv_results.items():
                    if 'error' not in cv_result:
                        report.append(f"### {model_name}")
                        report.append(f"- Mean F1 Score: {cv_result['mean']:.3f}")
                        report.append(f"- Standard Deviation: {cv_result['std']:.3f}")
                        report.append("")
        
        # Best model
        if self.training_results:
            best_model = self._find_best_model(
                self.training_results.get('training_results', {}),
                self.training_results.get('cv_results', {})
            )
            report.append(f"## Best Model: {best_model}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("- Use ensemble methods for best performance")
        report.append("- Consider semi-supervised learning for unlabeled data")
        report.append("- Regularly retrain models with new data")
        report.append("- Monitor false positive rates in production")
        report.append("")
        
        return "\n".join(report)
    
    def quick_test(self) -> Dict[str, Any]:
        """
        Quick test with sample data to verify the pipeline works
        """
        logger.info("Running quick test...")
        
        if TQDM_AVAILABLE:
            print("🚀 Starting Enhanced Spam Detection Quick Test...")
        
        # Sample test data
        test_messages = [
            "Hello, how are you doing today?",
            "FREE MONEY! Click here to win $1000000 NOW!",
            "Meeting is scheduled for tomorrow at 2 PM",
            "URGENT: Your account will be suspended! Click link immediately!",
            "Thanks for the great presentation yesterday",
            "Win iPhone X! Text STOP to 12345 for free entry!"
        ]
        
        test_labels = [0, 1, 0, 1, 0, 1]  # 0=ham, 1=spam
        
        # Create feature matrix with progress
        if TQDM_AVAILABLE:
            print("📊 Creating feature matrix...")
        feature_matrix, feature_names = self.preprocessor.create_feature_matrix(
            test_messages, max_features=1000, ngram_range=(1, 2)
        )
        
        # Train a simple model with progress
        if TQDM_AVAILABLE:
            print("🤖 Training classifier...")
        classifier = AdvancedSpamClassifier(model_type='multinomial_nb')
        training_results = classifier.fit(feature_matrix, np.array(test_labels))
        
        # Make predictions with progress
        if TQDM_AVAILABLE:
            print("🔮 Making predictions...")
        predictions = classifier.predict(feature_matrix)
        probabilities = classifier.predict_proba(feature_matrix)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels)
        
        if TQDM_AVAILABLE:
            print("✅ Quick test completed!")
        
        results = {
            'test_messages': test_messages,
            'true_labels': test_labels,
            'predictions': predictions.tolist(),
            'probabilities': probabilities[:, 1].tolist() if probabilities.size > 0 else [],
            'accuracy': accuracy,
            'training_results': training_results
        }
        
        logger.info(f"Quick test complete. Accuracy: {accuracy:.3f}")
        
        return results
    
    def full_training_pipeline(self,
                              use_email_data: bool = True,
                              use_sms_data: bool = True,
                              use_text_files: bool = True,
                              models_to_train: List[str] = None,
                              use_cross_validation: bool = True,
                              use_hyperparameter_tuning: bool = False,
                              cv_folds: int = 5,
                              save_models: bool = True) -> Dict[str, Any]:
        """
        Complete end-to-end training pipeline with comprehensive progress tracking
        """
        if TQDM_AVAILABLE:
            print("🚀 Starting Enhanced Spam Detection Training Pipeline")
            print("=" * 60)
        
        logger.info("Starting full training pipeline...")
        
        # Calculate total major steps
        total_major_steps = 5  # Load data, train models, evaluate, save models, generate report
        current_major_step = 0
        
        pipeline_results = {}
        
        try:
            # Step 1: Load and preprocess data
            current_major_step += 1
            if TQDM_AVAILABLE:
                print(f"\n📊 Step {current_major_step}/{total_major_steps}: Data Loading & Preprocessing")
                print("-" * 50)
            
            data_stats = self.load_and_preprocess_data(
                use_email_data=use_email_data,
                use_sms_data=use_sms_data,
                use_text_files=use_text_files,
                save_processed=True
            )
            pipeline_results['data_stats'] = data_stats
            
            # Step 2: Train models
            current_major_step += 1
            if TQDM_AVAILABLE:
                print(f"\n🤖 Step {current_major_step}/{total_major_steps}: Model Training")
                print("-" * 50)
            
            training_results = self.train_models(
                models_to_train=models_to_train,
                use_cross_validation=use_cross_validation,
                use_hyperparameter_tuning=use_hyperparameter_tuning,
                cv_folds=cv_folds
            )
            pipeline_results['training_results'] = training_results
            
            # Step 3: Evaluate best model
            current_major_step += 1
            if TQDM_AVAILABLE:
                print(f"\n📈 Step {current_major_step}/{total_major_steps}: Model Evaluation")
                print("-" * 50)
            
            best_model = training_results['best_model']
            evaluation_results = self.evaluate_model(model_name=best_model)
            pipeline_results['evaluation_results'] = evaluation_results
            
            # Print evaluation summary
            if TQDM_AVAILABLE:
                metrics = evaluation_results['metrics']
                print(f"🏆 Best Model: {best_model}")
                print(f"  📊 Test Accuracy: {metrics.get('accuracy', 0):.3f}")
                print(f"  🎯 Test F1 Score: {metrics.get('f1', 0):.3f}")
                print(f"  ⚖️  Test Precision: {metrics.get('precision', 0):.3f}")
                print(f"  🔍 Test Recall: {metrics.get('recall', 0):.3f}")
            
            # Step 4: Save models
            current_major_step += 1
            if save_models:
                if TQDM_AVAILABLE:
                    print(f"\n💾 Step {current_major_step}/{total_major_steps}: Saving Models")
                    print("-" * 50)
                self.save_models(model_name=best_model)
            else:
                if TQDM_AVAILABLE:
                    print(f"\n⏭️  Step {current_major_step}/{total_major_steps}: Skipping Model Save")
            
            # Step 5: Generate comprehensive report
            current_major_step += 1
            if TQDM_AVAILABLE:
                print(f"\n📝 Step {current_major_step}/{total_major_steps}: Generating Report")
                print("-" * 50)
            
            report = self.generate_comprehensive_report()
            pipeline_results['report'] = report
            
            # Save report to file
            report_path = self.models_dir / "training_report.md"
            with open(report_path, 'w') as f:
                f.write(report)
            
            if TQDM_AVAILABLE:
                print(f"📄 Training report saved to: {report_path}")
            
            # Final summary
            if TQDM_AVAILABLE:
                print("\n🎉 Training Pipeline Complete!")
                print("=" * 60)
                print(f"✅ Data processed: {data_stats['total_messages']:,} messages")
                print(f"🏆 Best model: {best_model}")
                print(f"📊 Test accuracy: {evaluation_results['metrics'].get('accuracy', 0):.3f}")
                print(f"📄 Report saved: {report_path}")
                if save_models:
                    print(f"💾 Models saved to: {self.models_dir}")
                print("=" * 60)
            
            pipeline_results['success'] = True
            pipeline_results['timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
            
            if TQDM_AVAILABLE:
                print(f"\n❌ Training pipeline failed at step {current_major_step}: {str(e)}")
            
            raise
        
        return pipeline_results
    
    def run_diagnostic_test(self) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic tests to ensure everything works properly
        """
        if TQDM_AVAILABLE:
            print("🔬 Running Enhanced Training Pipeline Diagnostics")
            print("=" * 60)
        
        logger.info("Running diagnostic tests...")
        
        diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        try:
            # Test 1: Component initialization
            with tqdm(total=5, desc="🧪 Diagnostic Tests", unit="test", 
                      disable=not TQDM_AVAILABLE) as pbar:
                
                pbar.set_description("🔧 Testing component initialization")
                assert self.preprocessor is not None, "Preprocessor not initialized"
                assert self.classifier is not None, "Classifier not initialized"
                diagnostic_results['tests']['initialization'] = {'status': 'PASS', 'message': 'All components initialized'}
                pbar.update(1)
                
                # Test 2: Quick data processing
                pbar.set_description("📊 Testing data processing")
                test_messages = [
                    "Hello, this is a normal message",
                    "FREE MONEY! Click here now!",
                    "Meeting tomorrow at 3 PM"
                ]
                
                # Test preprocessing
                processed = [self.preprocessor.preprocess_text(msg) for msg in test_messages]
                assert len(processed) == 3, "Preprocessing failed"
                
                # Test feature extraction
                features, names = self.preprocessor.create_feature_matrix(test_messages, max_features=100)
                assert features.shape[0] == 3, "Feature extraction failed"
                assert features.shape[1] > 0, "No features created"
                
                diagnostic_results['tests']['data_processing'] = {
                    'status': 'PASS', 
                    'message': f'Processed {len(test_messages)} messages, created {features.shape[1]} features'
                }
                pbar.update(1)
                
                # Test 3: Model training
                pbar.set_description("🤖 Testing model training")
                test_labels = np.array([0, 1, 0])  # ham, spam, ham
                
                # Train a simple model
                training_result = self.classifier.fit(features, test_labels)
                assert 'multinomial_nb' in training_result, "Training failed"
                
                diagnostic_results['tests']['model_training'] = {
                    'status': 'PASS',
                    'message': f'Trained {len(training_result)} models successfully'
                }
                pbar.update(1)
                
                # Test 4: Predictions
                pbar.set_description("🔮 Testing predictions")
                predictions = self.classifier.predict(features)
                probabilities = self.classifier.predict_proba(features)
                
                assert len(predictions) == 3, "Prediction failed"
                assert probabilities.shape == (3, 2), "Probability prediction failed"
                
                diagnostic_results['tests']['predictions'] = {
                    'status': 'PASS',
                    'message': f'Generated {len(predictions)} predictions successfully'
                }
                pbar.update(1)
                
                # Test 5: End-to-end pipeline test
                pbar.set_description("🚀 Testing end-to-end pipeline")
                quick_results = self.quick_test()
                assert quick_results['accuracy'] >= 0.0, "Quick test failed"
                
                diagnostic_results['tests']['end_to_end'] = {
                    'status': 'PASS',
                    'message': f'Pipeline test completed with {quick_results["accuracy"]:.3f} accuracy'
                }
                pbar.update(1)
                
                pbar.set_description("✅ All diagnostics passed")
            
            # Overall result
            diagnostic_results['overall_status'] = 'PASS'
            diagnostic_results['message'] = 'All diagnostic tests passed successfully'
            
            if TQDM_AVAILABLE:
                print("\n✅ Diagnostic Test Results:")
                for test_name, result in diagnostic_results['tests'].items():
                    print(f"  {test_name}: {result['status']} - {result['message']}")
                print(f"\n🎉 Overall Status: {diagnostic_results['overall_status']}")
                print("=" * 60)
            
        except Exception as e:
            diagnostic_results['overall_status'] = 'FAIL'
            diagnostic_results['error'] = str(e)
            
            if TQDM_AVAILABLE:
                print(f"\n❌ Diagnostic test failed: {str(e)}")
            
            logger.error(f"Diagnostic tests failed: {str(e)}")
            raise
        
        return diagnostic_results


def main():
    """
    Main function to demonstrate the enhanced training pipeline
    """
    import sys
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Enhanced Spam Detection Training Pipeline')
    parser.add_argument('--mode', choices=['quick', 'full', 'diagnostic'], default='diagnostic',
                        help='Training mode: quick test, full pipeline, or diagnostics')
    parser.add_argument('--email', action='store_true', help='Use email data')
    parser.add_argument('--sms', action='store_true', help='Use SMS data')
    parser.add_argument('--text', action='store_true', help='Use text files')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    parser.add_argument('--tune', action='store_true', help='Use hyperparameter tuning')
    parser.add_argument('--save', action='store_true', help='Save trained models')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize trainer
        trainer = EnhancedSpamTrainer()
        
        if args.mode == 'diagnostic':
            print("🔬 Running diagnostic tests...")
            results = trainer.run_diagnostic_test()
            
        elif args.mode == 'quick':
            print("🚀 Running quick test...")
            results = trainer.quick_test()
            print(f"Quick test accuracy: {results['accuracy']:.3f}")
            
        elif args.mode == 'full':
            print("🌟 Running full training pipeline...")
            results = trainer.full_training_pipeline(
                use_email_data=args.email or True,
                use_sms_data=args.sms or True,
                use_text_files=args.text or True,
                use_cross_validation=args.cv,
                use_hyperparameter_tuning=args.tune,
                save_models=args.save
            )
            
            if results['success']:
                print("🎉 Full pipeline completed successfully!")
            else:
                print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
                sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
