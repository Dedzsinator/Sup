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
        Load and preprocess all available data
        """
        logger.info("Loading and preprocessing data...")
        
        all_messages = []
        all_labels = []
        all_metadata = []
        
        # Load different data sources
        if use_email_data:
            email_messages, email_labels, email_metadata = self.preprocessor.load_email_data()
            all_messages.extend(email_messages)
            all_labels.extend(email_labels)
            all_metadata.extend(email_metadata)
            logger.info(f"Loaded {len(email_messages)} email messages")
        
        if use_sms_data:
            sms_messages, sms_labels, sms_metadata = self.preprocessor.load_sms_data()
            all_messages.extend(sms_messages)
            all_labels.extend(sms_labels)
            all_metadata.extend(sms_metadata)
            logger.info(f"Loaded {len(sms_messages)} SMS messages")
        
        if use_text_files:
            text_messages, text_labels, text_metadata = self.preprocessor.load_text_files()
            all_messages.extend(text_messages)
            all_labels.extend(text_labels)
            all_metadata.extend(text_metadata)
            logger.info(f"Loaded {len(text_messages)} text file messages")
        
        # Store training data
        self.training_data = {
            'messages': all_messages,
            'labels': np.array(all_labels),
            'metadata': all_metadata
        }
        
        # Create feature matrix
        logger.info("Creating feature matrix...")
        self.feature_matrix, self.feature_names = self.preprocessor.create_feature_matrix(
            all_messages, 
            max_features=5000,
            ngram_range=(1, 2),
            use_tfidf=True
        )
        
        # Save processed data if requested
        if save_processed:
            self.preprocessor.save_processed_data(
                all_messages, all_labels, all_metadata,
                filename="enhanced_processed_data.jsonl"
            )
        
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
        
        # Train models
        training_results = self.classifier.fit(
            self.feature_matrix, 
            self.training_data['labels'],
            semi_supervised_data=None  # Will be handled automatically
        )
        
        # Cross-validation
        cv_results = {}
        if use_cross_validation:
            logger.info("Performing cross-validation...")
            cv_results = self.classifier.cross_validate(
                self.feature_matrix,
                self.training_data['labels'],
                cv=cv_folds,
                scoring='f1'
            )
        
        # Hyperparameter tuning
        tuning_results = {}
        if use_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            for model_name in ['random_forest', 'logistic']:
                if model_name in models_to_train:
                    tuning_results[model_name] = self.classifier.hyperparameter_tuning(
                        self.feature_matrix,
                        self.training_data['labels'],
                        model_name=model_name,
                        cv=3,
                        n_iter=10
                    )
        
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
        Evaluate model performance on held-out test set
        """
        if self.training_data is None or self.feature_matrix is None:
            raise ValueError("Data must be loaded and preprocessed first")
        
        # Use only labeled data for evaluation
        labeled_mask = self.training_data['labels'] >= 0
        X_labeled = self.feature_matrix[labeled_mask]
        y_labeled = self.training_data['labels'][labeled_mask]
        
        # Split data
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
        
        # Train on training set
        self.classifier.fit(X_train, y_train)
        
        # Find best model if not specified
        if model_name is None:
            model_name = self._find_best_model(self.training_results.get('training_results', {}), 
                                             self.training_results.get('cv_results', {}))
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test, model_name=model_name)
        y_pred_proba = self.classifier.predict_proba(X_test, model_name=model_name)
        
        # Calculate metrics
        metrics = self.classifier._calculate_metrics(
            y_test, y_pred, 
            y_pred_proba[:, 1] if y_pred_proba.size > 0 else None
        )
        
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
        
        # Save the classifier
        model_path = self.models_dir / f"{model_name}_classifier.pkl"
        self.classifier.save_model(str(model_path), model_name)
        
        # Save the preprocessor
        preprocessor_path = self.models_dir / "preprocessor.pkl"
        import pickle
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save training results
        results_path = self.models_dir / "training_results.json"
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
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved:")
        logger.info(f"  Classifier: {model_path}")
        logger.info(f"  Preprocessor: {preprocessor_path}")
        logger.info(f"  Metadata: {metadata_path}")
    
    def load_models(self, model_name: str = None):
        """Load trained models from disk"""
        # Load metadata
        metadata_path = self.models_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if model_name is None:
                model_name = metadata.get('best_model', 'ensemble')
        
        # Load classifier
        model_path = self.models_dir / f"{model_name}_classifier.pkl"
        if model_path.exists():
            self.classifier.load_model(str(model_path))
        
        # Load preprocessor
        preprocessor_path = self.models_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            import pickle
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        
        # Load training results
        results_path = self.models_dir / "training_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                self.training_results = json.load(f)
        
        logger.info(f"Models loaded: {model_name}")
    
    def predict(self, messages: List[str], model_name: str = None) -> Dict[str, Any]:
        """
        Make predictions on new messages
        """
        if not messages:
            return {'predictions': [], 'probabilities': []}
        
        # Preprocess messages
        processed_messages = [self.preprocessor.preprocess_text(msg) for msg in messages]
        
        # Create feature matrix
        if hasattr(self.preprocessor, 'tfidf_vectorizer') and self.preprocessor.tfidf_vectorizer is not None:
            # Use existing vectorizer
            tfidf_features = self.preprocessor.tfidf_vectorizer.transform(processed_messages)
            
            # Extract other features
            linguistic_features = []
            spam_features = []
            
            for msg in messages:
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
        
        # Create feature matrix
        feature_matrix, feature_names = self.preprocessor.create_feature_matrix(
            test_messages, max_features=1000, ngram_range=(1, 2)
        )
        
        # Train a simple model
        classifier = AdvancedSpamClassifier(model_type='multinomial_nb')
        training_results = classifier.fit(feature_matrix, np.array(test_labels))
        
        # Make predictions
        predictions = classifier.predict(feature_matrix)
        probabilities = classifier.predict_proba(feature_matrix)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels)
        
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
