"""
Training Pipeline for Hyperoptimized Spam Detection

This module provides a comprehensive training pipeline for the spam detection system,
including data preprocessing, model training, evaluation, and optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from classifier import HyperoptimizedBayesClassifier

logger = logging.getLogger(__name__)

class SpamDetectionTrainer:
    """
    Training pipeline for spam detection system
    """
    
    def __init__(self, 
                 data_dir: str = "/app/data",
                 models_dir: str = "/app/models",
                 config_dir: str = "/app/config"):
        
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.classifier = None
        self.training_history = []
        
        logger.info(f"Initialized trainer with data_dir={data_dir}")
    
    def load_training_data(self, dataset_name: str = "spam_dataset") -> Tuple[List[str], List[bool], List[str], List[datetime]]:
        """
        Load training data from various sources
        
        Returns:
            Tuple of (messages, labels, user_ids, timestamps)
        """
        
        dataset_path = self.data_dir / f"{dataset_name}.jsonl"
        
        if not dataset_path.exists():
            logger.warning(f"Dataset {dataset_path} not found, generating synthetic data")
            return self._generate_synthetic_data()
        
        messages = []
        labels = []
        user_ids = []
        timestamps = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    messages.append(data['message'])
                    labels.append(data['is_spam'])
                    user_ids.append(data.get('user_id', 'unknown'))
                    
                    # Parse timestamp or use current time
                    if 'timestamp' in data:
                        timestamp = datetime.fromisoformat(data['timestamp'])
                    else:
                        timestamp = datetime.now()
                    timestamps.append(timestamp)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid line: {e}")
                    continue
        
        logger.info(f"Loaded {len(messages)} messages from {dataset_path}")
        return messages, labels, user_ids, timestamps
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[List[str], List[bool], List[str], List[datetime]]:
        """Generate synthetic spam/ham data for testing"""
        
        # Spam message templates
        spam_templates = [
            "URGENT! Win $1000 now! Click here: {url}",
            "Free iPhone! Limited time offer! Text {phone} now!",
            "Make money fast! Work from home! {email}",
            "Your account will be suspended! Click {url} immediately!",
            "CONGRATULATIONS! You've won the lottery! Call {phone}",
            "Cheap medications online! No prescription needed! {url}",
            "Hot singles in your area! Meet them now! {url}",
            "Lose weight fast! Amazing pills! Order at {url}",
            "Your computer is infected! Download cleaner: {url}",
            "Nigerian prince needs help! Reward $10M! Email {email}"
        ]
        
        # Ham message templates
        ham_templates = [
            "Hey, how are you doing today?",
            "Can we meet for lunch tomorrow?",
            "Thanks for the help with the project!",
            "The meeting is scheduled for 2 PM",
            "Happy birthday! Hope you have a great day!",
            "Did you see the game last night?",
            "Can you send me the report when you get a chance?",
            "The weather is really nice today",
            "Looking forward to seeing you soon",
            "Great job on the presentation!"
        ]
        
        messages = []
        labels = []
        user_ids = []
        timestamps = []
        
        # Generate synthetic users
        users = [f"user_{i}" for i in range(100)]
        
        # Generate spam messages (30% of dataset)
        spam_count = int(n_samples * 0.3)
        for i in range(spam_count):
            template = np.random.choice(spam_templates)
            message = template.format(
                url="http://spam-site.com",
                phone="555-SPAM",
                email="spam@evil.com"
            )
            
            messages.append(message)
            labels.append(True)  # Spam
            user_ids.append(np.random.choice(users))
            
            # Random timestamp within last 30 days
            days_ago = np.random.randint(0, 30)
            timestamp = datetime.now() - timedelta(days=days_ago)
            timestamps.append(timestamp)
        
        # Generate ham messages (70% of dataset)
        ham_count = n_samples - spam_count
        for i in range(ham_count):
            template = np.random.choice(ham_templates)
            
            messages.append(template)
            labels.append(False)  # Ham
            user_ids.append(np.random.choice(users))
            
            # Random timestamp within last 30 days
            days_ago = np.random.randint(0, 30)
            timestamp = datetime.now() - timedelta(days=days_ago)
            timestamps.append(timestamp)
        
        # Shuffle the data
        combined = list(zip(messages, labels, user_ids, timestamps))
        np.random.shuffle(combined)
        messages, labels, user_ids, timestamps = zip(*combined)
        
        logger.info(f"Generated {len(messages)} synthetic messages ({spam_count} spam, {ham_count} ham)")
        return list(messages), list(labels), list(user_ids), list(timestamps)
    
    def train_model(self, 
                   messages: List[str], 
                   labels: List[bool], 
                   user_ids: List[str], 
                   timestamps: List[datetime],
                   test_size: float = 0.2,
                   vocab_size: int = 50000,
                   user_vector_dim: int = 512) -> Dict:
        """
        Train the spam detection model
        
        Returns:
            Dictionary with training results and metrics
        """
        
        logger.info(f"Starting training with {len(messages)} messages")
        
        # Split data into train/test
        train_messages, test_messages, train_labels, test_labels, \
        train_users, test_users, train_timestamps, test_timestamps = train_test_split(
            messages, labels, user_ids, timestamps,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
        
        # Initialize classifier
        self.classifier = HyperoptimizedBayesClassifier(
            vocab_size=vocab_size,
            user_vector_dim=user_vector_dim
        )
        
        # Train the model
        start_time = datetime.now()
        self.classifier.train(train_messages, train_labels, train_users, train_timestamps)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate on test set
        test_results = self._evaluate_model(
            test_messages, test_labels, test_users, test_timestamps
        )
        
        # Cross-validation for more robust evaluation
        cv_results = self._cross_validate(messages, labels, user_ids, timestamps)
        
        # Prepare results
        results = {
            'training_time_seconds': training_time,
            'train_size': len(train_messages),
            'test_size': len(test_messages),
            'test_metrics': test_results,
            'cv_metrics': cv_results,
            'model_stats': self.classifier.get_model_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_history.append(results)
        
        # Save model and results
        self._save_training_results(results)
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Test accuracy: {test_results['accuracy']:.3f}")
        logger.info(f"Test F1: {test_results['f1']:.3f}")
        
        return results
    
    def _evaluate_model(self, messages: List[str], labels: List[bool], 
                       user_ids: List[str], timestamps: List[datetime]) -> Dict:
        """Evaluate model performance on test data"""
        
        predictions = []
        confidences = []
        
        for message, user_id, timestamp in zip(messages, user_ids, timestamps):
            spam_prob, confidence = self.classifier.predict(message, user_id, timestamp)
            predictions.append(spam_prob > 0.5)
            confidences.append(confidence)
        
        # Calculate metrics
        spam_probs = [self.classifier.predict(msg, uid, ts)[0] 
                     for msg, uid, ts in zip(messages, user_ids, timestamps)]
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'roc_auc': roc_auc_score(labels, spam_probs),
            'avg_confidence': np.mean(confidences)
        }
        
        return metrics
    
    def _cross_validate(self, messages: List[str], labels: List[bool],
                       user_ids: List[str], timestamps: List[datetime],
                       cv_folds: int = 5) -> Dict:
        """Perform cross-validation for robust evaluation"""
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Convert to numpy arrays for easier indexing
        messages_np = np.array(messages)
        labels_np = np.array(labels)
        users_np = np.array(user_ids)
        timestamps_np = np.array(timestamps)
        
        # Stratified K-fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(messages_np, labels_np)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Split data for this fold
            train_msgs = messages_np[train_idx].tolist()
            train_lbls = labels_np[train_idx].tolist()
            train_usrs = users_np[train_idx].tolist()
            train_ts = timestamps_np[train_idx].tolist()
            
            val_msgs = messages_np[val_idx].tolist()
            val_lbls = labels_np[val_idx].tolist()
            val_usrs = users_np[val_idx].tolist()
            val_ts = timestamps_np[val_idx].tolist()
            
            # Train a new classifier for this fold
            fold_classifier = HyperoptimizedBayesClassifier(
                vocab_size=self.classifier.vocab_size,
                user_vector_dim=self.classifier.user_vector_dim
            )
            fold_classifier.train(train_msgs, train_lbls, train_usrs, train_ts)
            
            # Evaluate on validation set
            val_predictions = []
            val_probs = []
            
            for msg, user, ts in zip(val_msgs, val_usrs, val_ts):
                spam_prob, _ = fold_classifier.predict(msg, user, ts)
                val_predictions.append(spam_prob > 0.5)
                val_probs.append(spam_prob)
            
            # Calculate metrics for this fold
            fold_metrics = {
                'accuracy': accuracy_score(val_lbls, val_predictions),
                'precision': precision_score(val_lbls, val_predictions, zero_division=0),
                'recall': recall_score(val_lbls, val_predictions, zero_division=0),
                'f1': f1_score(val_lbls, val_predictions, zero_division=0),
                'roc_auc': roc_auc_score(val_lbls, val_probs)
            }
            
            for metric, score in fold_metrics.items():
                cv_scores[metric].append(score)
        
        # Calculate mean and std for each metric
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        return cv_results
    
    def _save_training_results(self, results: Dict) -> None:
        """Save training results and model"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_dir / f"spam_classifier_{timestamp}.pkl"
        self.classifier.save_model(str(model_path))
        
        # Save results
        results_path = self.config_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save training history
        history_path = self.config_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info(f"Saved model to {model_path}")
        logger.info(f"Saved results to {results_path}")
    
    def load_best_model(self) -> Optional[str]:
        """Load the best performing model"""
        
        if not self.training_history:
            # Try to load history from file
            history_path = self.config_dir / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
        
        if not self.training_history:
            logger.warning("No training history found")
            return None
        
        # Find best model by F1 score
        best_result = max(self.training_history, 
                         key=lambda x: x['test_metrics']['f1'])
        
        # Find corresponding model file
        timestamp = datetime.fromisoformat(best_result['timestamp']).strftime("%Y%m%d_%H%M%S")
        model_path = self.models_dir / f"spam_classifier_{timestamp}.pkl"
        
        if model_path.exists():
            self.classifier = HyperoptimizedBayesClassifier()
            self.classifier.load_model(str(model_path))
            logger.info(f"Loaded best model from {model_path}")
            return str(model_path)
        else:
            logger.warning(f"Best model file not found: {model_path}")
            return None
    
    def generate_training_report(self) -> str:
        """Generate a comprehensive training report"""
        
        if not self.training_history:
            return "No training history available"
        
        latest_result = self.training_history[-1]
        
        report = f"""
# Spam Detection Training Report

## Model Performance

### Latest Training Results
- **Training Time**: {latest_result['training_time_seconds']:.2f} seconds
- **Training Set Size**: {latest_result['train_size']} messages
- **Test Set Size**: {latest_result['test_size']} messages

### Test Set Metrics
- **Accuracy**: {latest_result['test_metrics']['accuracy']:.3f}
- **Precision**: {latest_result['test_metrics']['precision']:.3f}
- **Recall**: {latest_result['test_metrics']['recall']:.3f}
- **F1 Score**: {latest_result['test_metrics']['f1']:.3f}
- **ROC AUC**: {latest_result['test_metrics']['roc_auc']:.3f}
- **Average Confidence**: {latest_result['test_metrics']['avg_confidence']:.3f}

### Cross-Validation Results (5-fold)
- **Accuracy**: {latest_result['cv_metrics']['accuracy_mean']:.3f} ± {latest_result['cv_metrics']['accuracy_std']:.3f}
- **Precision**: {latest_result['cv_metrics']['precision_mean']:.3f} ± {latest_result['cv_metrics']['precision_std']:.3f}
- **Recall**: {latest_result['cv_metrics']['recall_mean']:.3f} ± {latest_result['cv_metrics']['recall_std']:.3f}
- **F1 Score**: {latest_result['cv_metrics']['f1_mean']:.3f} ± {latest_result['cv_metrics']['f1_std']:.3f}
- **ROC AUC**: {latest_result['cv_metrics']['roc_auc_mean']:.3f} ± {latest_result['cv_metrics']['roc_auc_std']:.3f}

### Model Statistics
- **Total Messages Trained**: {latest_result['model_stats']['total_messages_trained']}
- **Spam Messages**: {latest_result['model_stats']['spam_messages']}
- **Ham Messages**: {latest_result['model_stats']['ham_messages']}
- **Vocabulary Size**: {latest_result['model_stats']['vocab_size']}
- **User Profiles**: {latest_result['model_stats']['user_profiles']}

## Training History
Total training runs: {len(self.training_history)}

"""
        
        # Add performance trends if multiple training runs
        if len(self.training_history) > 1:
            f1_scores = [r['test_metrics']['f1'] for r in self.training_history]
            accuracy_scores = [r['test_metrics']['accuracy'] for r in self.training_history]
            
            report += f"""
### Performance Trends
- **Best F1 Score**: {max(f1_scores):.3f}
- **Best Accuracy**: {max(accuracy_scores):.3f}
- **Latest F1**: {f1_scores[-1]:.3f}
- **Latest Accuracy**: {accuracy_scores[-1]:.3f}
"""
        
        return report
    
    def hyperparameter_optimization(self, messages: List[str], labels: List[bool],
                                  user_ids: List[str], timestamps: List[datetime]) -> Dict:
        """
        Perform hyperparameter optimization
        """
        
        logger.info("Starting hyperparameter optimization")
        
        # Define hyperparameter search space
        param_grid = {
            'vocab_size': [10000, 25000, 50000],
            'user_vector_dim': [128, 256, 512],
            'temporal_decay': [0.9, 0.95, 0.99],
            'confidence_threshold': [0.7, 0.8, 0.9]
        }
        
        best_score = 0
        best_params = None
        results = []
        
        # Grid search (simplified version)
        for vocab_size in param_grid['vocab_size']:
            for user_vector_dim in param_grid['user_vector_dim']:
                for temporal_decay in param_grid['temporal_decay']:
                    for confidence_threshold in param_grid['confidence_threshold']:
                        
                        params = {
                            'vocab_size': vocab_size,
                            'user_vector_dim': user_vector_dim,
                            'temporal_decay': temporal_decay,
                            'confidence_threshold': confidence_threshold
                        }
                        
                        logger.info(f"Testing params: {params}")
                        
                        # Cross-validation with these parameters
                        cv_score = self._evaluate_hyperparams(
                            messages, labels, user_ids, timestamps, params
                        )
                        
                        results.append({
                            'params': params,
                            'cv_score': cv_score
                        })
                        
                        if cv_score > best_score:
                            best_score = cv_score
                            best_params = params
                            
                        logger.info(f"CV F1 score: {cv_score:.3f}")
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best CV score: {best_score:.3f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def _evaluate_hyperparams(self, messages: List[str], labels: List[bool],
                             user_ids: List[str], timestamps: List[datetime],
                             params: Dict) -> float:
        """Evaluate hyperparameters using cross-validation"""
        
        # Simple 3-fold CV for speed
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        messages_np = np.array(messages)
        labels_np = np.array(labels)
        users_np = np.array(user_ids)
        timestamps_np = np.array(timestamps)
        
        for train_idx, val_idx in kf.split(messages_np, labels_np):
            # Create classifier with these hyperparameters
            classifier = HyperoptimizedBayesClassifier(**params)
            
            # Train
            train_msgs = messages_np[train_idx].tolist()
            train_lbls = labels_np[train_idx].tolist()
            train_usrs = users_np[train_idx].tolist()
            train_ts = timestamps_np[train_idx].tolist()
            
            classifier.train(train_msgs, train_lbls, train_usrs, train_ts)
            
            # Validate
            val_msgs = messages_np[val_idx].tolist()
            val_lbls = labels_np[val_idx].tolist()
            val_usrs = users_np[val_idx].tolist()
            val_ts = timestamps_np[val_idx].tolist()
            
            predictions = []
            for msg, user, ts in zip(val_msgs, val_usrs, val_ts):
                spam_prob, _ = classifier.predict(msg, user, ts)
                predictions.append(spam_prob > 0.5)
            
            f1 = f1_score(val_lbls, predictions, zero_division=0)
            scores.append(f1)
        
        return np.mean(scores)
