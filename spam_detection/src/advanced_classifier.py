"""
Advanced Machine Learning Models for Spam Detection

This module implements state-of-the-art spam detection algorithms including:
- Naive Bayes with advanced feature engineering
- Semi-supervised learning techniques
- Ensemble methods
- Cross-validation with hyperparameter optimization
- Model persistence and deployment features
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import warnings

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a dummy tqdm class
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable or []
            self.desc = desc
            self.total = total
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass
        
        def set_description(self, desc):
            pass
        
        def close(self):
            pass

# Core ML imports
try:
    from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        cross_val_score, GridSearchCV, RandomizedSearchCV, 
        StratifiedKFold, train_test_split
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix,
        roc_curve, precision_recall_curve
    )
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Advanced ML features will be limited.")

# Optional advanced imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy import stats
    import scipy.sparse as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedSpamClassifier:
    """
    Advanced spam detection classifier with multiple algorithms and techniques
    """
    
    def __init__(self, 
                 model_type: str = "ensemble",
                 enable_semi_supervised: bool = True,
                 enable_feature_selection: bool = True,
                 random_state: int = 42):
        
        self.model_type = model_type
        self.enable_semi_supervised = enable_semi_supervised
        self.enable_feature_selection = enable_feature_selection
        self.random_state = random_state
        
        # Model storage
        self.models = {}
        self.feature_selector = None
        self.scaler = None
        self.dimensionality_reducer = None
        
        # Training history
        self.training_history = []
        self.cv_scores = {}
        self.feature_importance = None
        
        # Initialize models
        self._init_models()
        
        logger.info(f"Initialized AdvancedSpamClassifier with model_type={model_type}")
    
    def _init_models(self):
        """Initialize various machine learning models"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Using dummy classifier.")
            self.models['dummy'] = DummySpamClassifier()
            return
        
        # Naive Bayes variants
        self.models['multinomial_nb'] = MultinomialNB(alpha=1.0)
        self.models['complement_nb'] = ComplementNB(alpha=1.0)
        self.models['gaussian_nb'] = GaussianNB()
        
        # Other classifiers
        self.models['logistic'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'
        )
        
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=self.random_state
        )
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Ensemble model
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('nb', MultinomialNB(alpha=0.1)),
                ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state))
            ],
            voting='soft'
        )
        
        # Semi-supervised models
        if self.enable_semi_supervised:
            self.models['label_propagation'] = LabelPropagation(
                kernel='knn',
                n_neighbors=7,
                gamma=0.2
            )
            
            self.models['label_spreading'] = LabelSpreading(
                kernel='knn',
                n_neighbors=7,
                alpha=0.2
            )
    
    def prepare_features(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                        fit_transforms: bool = False) -> np.ndarray:
        """
        Prepare features with scaling, selection, and dimensionality reduction
        """
        X_processed = X.copy()
        
        # Handle sparse matrices
        if hasattr(X_processed, 'toarray'):
            is_sparse = True
            X_dense = X_processed.toarray()
        else:
            is_sparse = False
            X_dense = X_processed
        
        # Feature scaling for dense features (last columns are usually linguistic features)
        if SKLEARN_AVAILABLE and X_dense.shape[1] > 100:  # Assume TF-IDF + other features
            # Scale only the non-TF-IDF features (usually the last columns)
            tfidf_features = X_dense[:, :-20]  # Assume last 20 are linguistic/spam features
            other_features = X_dense[:, -20:]
            
            if fit_transforms:
                self.scaler = StandardScaler()
                other_features_scaled = self.scaler.fit_transform(other_features)
            elif self.scaler is not None:
                other_features_scaled = self.scaler.transform(other_features)
            else:
                other_features_scaled = other_features
            
            X_processed = np.hstack([tfidf_features, other_features_scaled])
        else:
            X_processed = X_dense
        
        # Feature selection
        if self.enable_feature_selection and SKLEARN_AVAILABLE and y is not None and fit_transforms:
            n_features = min(5000, X_processed.shape[1])  # Select top 5000 features
            self.feature_selector = SelectKBest(
                score_func=chi2 if np.all(X_processed >= 0) else mutual_info_classif,
                k=n_features
            )
            X_processed = self.feature_selector.fit_transform(X_processed, y)
        elif self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        # Dimensionality reduction for very high-dimensional data
        if X_processed.shape[1] > 10000 and SKLEARN_AVAILABLE:
            if fit_transforms:
                self.dimensionality_reducer = TruncatedSVD(
                    n_components=min(1000, X_processed.shape[1] - 1),
                    random_state=self.random_state
                )
                X_processed = self.dimensionality_reducer.fit_transform(X_processed)
            elif self.dimensionality_reducer is not None:
                X_processed = self.dimensionality_reducer.transform(X_processed)
        
        return X_processed
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            semi_supervised_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Train multiple models and return training results
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Using dummy training.")
            return {'dummy': {'accuracy': 0.5}}
        
        # Prepare features
        X_processed = self.prepare_features(X, y, fit_transforms=True)
        
        # Separate labeled and unlabeled data for semi-supervised learning
        labeled_mask = y >= 0
        X_labeled = X_processed[labeled_mask]
        y_labeled = y[labeled_mask]
        
        results = {}
        
        # Train supervised models
        supervised_models = ['multinomial_nb', 'complement_nb', 'logistic', 'random_forest', 'ensemble']
        
        logger.info(f"Training {len(supervised_models)} supervised models...")
        for model_name in tqdm(supervised_models, desc="Training models", unit="model"):
            if model_name not in self.models:
                continue
            
            try:
                logger.info(f"Training {model_name}...")
                
                # Handle different data requirements
                if model_name in ['multinomial_nb', 'complement_nb']:
                    # These need non-negative features
                    X_train = np.abs(X_labeled) if np.any(X_labeled < 0) else X_labeled
                else:
                    X_train = X_labeled
                
                # Train model
                self.models[model_name].fit(X_train, y_labeled)
                
                # Evaluate
                y_pred = self.models[model_name].predict(X_train)
                y_pred_proba = None
                if hasattr(self.models[model_name], 'predict_proba'):
                    y_pred_proba = self.models[model_name].predict_proba(X_train)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_labeled, y_pred, y_pred_proba)
                results[model_name] = metrics
                
                logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Train semi-supervised models if unlabeled data is available
        if self.enable_semi_supervised and np.any(y < 0):
            unlabeled_mask = y < 0
            X_unlabeled = X_processed[unlabeled_mask]
            
            # Create labels for semi-supervised learning (-1 for unlabeled)
            y_semi = y.copy()
            y_semi[unlabeled_mask] = -1
            
            semi_models = ['label_propagation', 'label_spreading']
            
            for model_name in semi_models:
                if model_name not in self.models:
                    continue
                
                try:
                    logger.info(f"Training {model_name} with {sum(unlabeled_mask)} unlabeled samples...")
                    
                    # Handle sparse matrices for semi-supervised learning
                    if hasattr(X_processed, 'toarray'):
                        X_semi = X_processed.toarray()
                    else:
                        X_semi = X_processed
                    
                    # Train semi-supervised model
                    self.models[model_name].fit(X_semi, y_semi)
                    
                    # Evaluate on labeled data only
                    y_pred_labeled = self.models[model_name].predict(X_labeled)
                    y_pred_proba_labeled = None
                    if hasattr(self.models[model_name], 'predict_proba'):
                        y_pred_proba_labeled = self.models[model_name].predict_proba(X_labeled)[:, 1]
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(y_labeled, y_pred_labeled, y_pred_proba_labeled)
                    results[model_name] = metrics
                    
                    logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_labeled),
            'n_features': X_processed.shape[1],
            'results': results
        })
        
        return results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'f1') -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on multiple models
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Skipping cross-validation.")
            return {}
        
        # Prepare features
        X_processed = self.prepare_features(X, y, fit_transforms=True)
        
        # Only use labeled data
        labeled_mask = y >= 0
        X_labeled = X_processed[labeled_mask]
        y_labeled = y[labeled_mask]
        
        cv_results = {}
        
        # Cross-validate supervised models
        supervised_models = ['multinomial_nb', 'complement_nb', 'logistic', 'random_forest']
        
        logger.info(f"Cross-validating {len(supervised_models)} models...")
        for model_name in tqdm(supervised_models, desc="Cross-validating models", unit="model"):
            if model_name not in self.models:
                continue
            
            try:
                logger.info(f"Cross-validating {model_name}...")
                
                model = self.models[model_name]
                
                # Handle different data requirements
                if model_name in ['multinomial_nb', 'complement_nb']:
                    X_cv = np.abs(X_labeled) if np.any(X_labeled < 0) else X_labeled
                else:
                    X_cv = X_labeled
                
                # Perform cross-validation
                cv_scores = cross_val_score(
                    model, X_cv, y_labeled, 
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                    scoring=scoring,
                    n_jobs=-1
                )
                
                cv_results[model_name] = {
                    'mean': np.mean(cv_scores),
                    'std': np.std(cv_scores),
                    'scores': cv_scores.tolist()
                }
                
                logger.info(f"{model_name} CV {scoring}: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
                
            except Exception as e:
                logger.error(f"Error in cross-validation for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        self.cv_scores = cv_results
        return cv_results
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             model_name: str = 'random_forest',
                             param_grid: Optional[Dict] = None,
                             cv: int = 3,
                             n_iter: int = 20) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Skipping hyperparameter tuning.")
            return {}
        
        # Prepare features
        X_processed = self.prepare_features(X, y, fit_transforms=True)
        
        # Only use labeled data
        labeled_mask = y >= 0
        X_labeled = X_processed[labeled_mask]
        y_labeled = y[labeled_mask]
        
        # Default parameter grids
        default_param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        }
        
        if param_grid is None:
            param_grid = default_param_grids.get(model_name, {})
        
        if model_name not in self.models or not param_grid:
            logger.warning(f"No parameter grid available for {model_name}")
            return {}
        
        try:
            logger.info(f"Hyperparameter tuning for {model_name}...")
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                self.models[model_name],
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=-1,
                random_state=self.random_state
            )
            
            # Handle different data requirements
            if model_name in ['multinomial_nb', 'complement_nb']:
                X_tune = np.abs(X_labeled) if np.any(X_labeled < 0) else X_labeled
            else:
                X_tune = X_labeled
            
            search.fit(X_tune, y_labeled)
            
            # Update the model with best parameters
            self.models[model_name] = search.best_estimator_
            
            results = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
            logger.info(f"Best {model_name} score: {search.best_score_:.3f}")
            logger.info(f"Best {model_name} params: {search.best_params_}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {e}")
            return {'error': str(e)}
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using the specified model or the best performing model
        """
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return np.array([])
        
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Handle different data requirements
        if model_name in ['multinomial_nb', 'complement_nb']:
            X_pred = np.abs(X_processed) if np.any(X_processed < 0) else X_processed
        else:
            X_pred = X_processed
        
        return self.models[model_name].predict(X_pred)
    
    def predict_proba(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Predict class probabilities
        """
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return np.array([])
        
        model = self.models[model_name]
        if not hasattr(model, 'predict_proba'):
            logger.warning(f"Model {model_name} does not support probability prediction")
            return np.array([])
        
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Handle different data requirements
        if model_name in ['multinomial_nb', 'complement_nb']:
            X_pred = np.abs(X_processed) if np.any(X_processed < 0) else X_processed
        else:
            X_pred = X_processed
        
        return model.predict_proba(X_pred)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save model to file"""
        if model_name is None:
            model_name = self.model_type
        
        model_data = {
            'model': self.models.get(model_name),
            'feature_selector': self.feature_selector,
            'scaler': self.scaler,
            'dimensionality_reducer': self.dimensionality_reducer,
            'model_type': model_name,
            'training_history': self.training_history,
            'cv_scores': self.cv_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[model_data['model_type']] = model_data['model']
        self.feature_selector = model_data.get('feature_selector')
        self.scaler = model_data.get('scaler')
        self.dimensionality_reducer = model_data.get('dimensionality_reducer')
        self.training_history = model_data.get('training_history', [])
        self.cv_scores = model_data.get('cv_scores', {})
        
        logger.info(f"Model loaded from {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive training report"""
        report = ["# Advanced Spam Detection Model Report\n"]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Training history
        if self.training_history:
            last_training = self.training_history[-1]
            report.append("## Last Training Session")
            report.append(f"- Timestamp: {last_training['timestamp']}")
            report.append(f"- Samples: {last_training['n_samples']}")
            report.append(f"- Features: {last_training['n_features']}")
            report.append("")
            
            # Model performance
            report.append("### Model Performance")
            for model_name, metrics in last_training['results'].items():
                if 'error' not in metrics:
                    report.append(f"**{model_name}:**")
                    report.append(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
                    report.append(f"- Precision: {metrics.get('precision', 0):.3f}")
                    report.append(f"- Recall: {metrics.get('recall', 0):.3f}")
                    report.append(f"- F1 Score: {metrics.get('f1', 0):.3f}")
                    if 'roc_auc' in metrics:
                        report.append(f"- ROC AUC: {metrics['roc_auc']:.3f}")
                    report.append("")
        
        # Cross-validation results
        if self.cv_scores:
            report.append("## Cross-Validation Results")
            for model_name, cv_result in self.cv_scores.items():
                if 'error' not in cv_result:
                    report.append(f"**{model_name}:**")
                    report.append(f"- Mean Score: {cv_result['mean']:.3f}")
                    report.append(f"- Std Dev: {cv_result['std']:.3f}")
                    report.append("")
        
        return "\n".join(report)


class DummySpamClassifier:
    """Dummy classifier for when sklearn is not available"""
    
    def __init__(self):
        self.is_fitted = False
        self.spam_keywords = [
            'free', 'win', 'winner', 'cash', 'money', 'prize', 'urgent',
            'click', 'offer', 'deal', 'viagra', 'pharmacy', 'loan'
        ]
    
    def fit(self, X, y):
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Simple rule-based prediction
        predictions = []
        for i in range(len(X)):
            # Assume X is feature matrix, use simple heuristics
            predictions.append(np.random.choice([0, 1]))  # Random for now
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        predictions = self.predict(X)
        probas = np.zeros((len(predictions), 2))
        for i, pred in enumerate(predictions):
            if pred == 1:
                probas[i] = [0.3, 0.7]  # Spam
            else:
                probas[i] = [0.7, 0.3]  # Ham
        return probas
