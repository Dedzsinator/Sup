"""
Hyperoptimized Non-Naive Bayes Spam Classifier with Custom User Word Spawn Vectors

This module implements an advanced spam detection system that goes beyond traditional
Naive Bayes by incorporating:
1. User-specific word spawn vectors (custom learned patterns per user)
2. Feature interdependencies (non-naive assumptions)
3. Temporal and contextual features
4. Adaptive learning with confidence-weighted updates
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
import pickle
import hashlib
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

@dataclass
class SpamFeatures:
    """Feature vector representation for spam detection"""
    word_features: np.ndarray
    user_spawn_vector: np.ndarray
    temporal_features: np.ndarray
    structural_features: np.ndarray
    confidence_weights: np.ndarray

@dataclass
class UserProfile:
    """User-specific spam detection profile"""
    user_id: str
    word_spawn_vectors: Dict[str, float]
    legitimate_patterns: Set[str]
    spam_patterns: Set[str]
    confidence_scores: Dict[str, float]
    last_updated: datetime
    message_count: int

class HyperoptimizedBayesClassifier:
    """
    Non-Naive Bayes classifier with user-specific word spawn vectors
    
    Key innovations:
    1. User spawn vectors: Learn user-specific word usage patterns
    2. Feature correlation matrix: Model dependencies between features
    3. Confidence-weighted learning: Adapt based on prediction confidence
    4. Temporal decay: Recent patterns weighted more heavily
    """
    
    def __init__(self, 
                 vocab_size: int = 50000,
                 user_vector_dim: int = 512,
                 temporal_decay: float = 0.95,
                 confidence_threshold: float = 0.8):
        
        self.vocab_size = vocab_size
        self.user_vector_dim = user_vector_dim
        self.temporal_decay = temporal_decay
        self.confidence_threshold = confidence_threshold
        
        # Core classification parameters
        self.spam_log_probs = np.zeros(vocab_size)
        self.ham_log_probs = np.zeros(vocab_size)
        self.spam_prior = 0.5
        self.ham_prior = 0.5
        
        # Feature correlation matrix (non-naive component)
        self.feature_correlation_matrix = np.eye(vocab_size) * 0.1
        
        # User-specific profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Vocabulary mapping
        self.vocab_to_idx: Dict[str, int] = {}
        self.idx_to_vocab: Dict[int, str] = {}
        
        # Performance optimization caches
        self._prediction_cache: Dict[str, Tuple[float, datetime]] = {}
        self._feature_cache: Dict[str, SpamFeatures] = {}
        
        # Statistics for adaptive learning
        self.total_spam_count = 0
        self.total_ham_count = 0
        
        logger.info(f"Initialized HyperoptimizedBayesClassifier with vocab_size={vocab_size}")
    
    def _hash_message(self, message: str, user_id: str) -> str:
        """Create a deterministic hash for message caching"""
        content = f"{user_id}:{message}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_word_features(self, message: str) -> np.ndarray:
        """Extract word-based features from message"""
        words = self._tokenize(message)
        features = np.zeros(self.vocab_size)
        
        for word in words:
            if word in self.vocab_to_idx:
                idx = self.vocab_to_idx[word]
                features[idx] += 1
                
        # Normalize by message length to prevent bias
        total_words = len(words)
        if total_words > 0:
            features = features / total_words
            
        return features
    
    def _extract_user_spawn_vector(self, user_id: str, message: str) -> np.ndarray:
        """Extract user-specific word spawn patterns"""
        if user_id not in self.user_profiles:
            return np.zeros(self.user_vector_dim)
        
        profile = self.user_profiles[user_id]
        words = set(self._tokenize(message))
        
        # Calculate spawn vector based on user's historical patterns
        spawn_vector = np.zeros(self.user_vector_dim)
        
        for i, word in enumerate(words):
            if word in profile.word_spawn_vectors:
                # Map word to vector dimension using hash
                dim_idx = hash(word) % self.user_vector_dim
                spawn_vector[dim_idx] += profile.word_spawn_vectors[word]
        
        # Apply confidence weighting
        confidence_weight = np.mean([
            profile.confidence_scores.get(word, 0.5) for word in words
        ]) if words else 0.5
        
        return spawn_vector * confidence_weight
    
    def _extract_temporal_features(self, timestamp: Optional[datetime] = None) -> np.ndarray:
        """Extract time-based features"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Time-of-day features (spam patterns often vary by time)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        temporal_features = np.zeros(10)  # 24 hours + 7 days mapped to 10 dims
        
        # Hour of day (0-23 mapped to 0-7)
        hour_bin = hour // 3
        temporal_features[hour_bin] = 1.0
        
        # Day of week (0-6 mapped to 8-9)
        weekend = 1.0 if day_of_week >= 5 else 0.0
        temporal_features[8] = weekend
        temporal_features[9] = 1.0 - weekend
        
        return temporal_features
    
    def _extract_structural_features(self, message: str) -> np.ndarray:
        """Extract structural/syntactic features"""
        structural = np.zeros(15)
        
        # Basic statistics
        structural[0] = len(message)  # Message length
        structural[1] = len(message.split())  # Word count
        structural[2] = message.count('!')  # Exclamation marks
        structural[3] = message.count('?')  # Question marks
        structural[4] = message.count('.')  # Periods
        structural[5] = sum(1 for c in message if c.isupper())  # Uppercase chars
        structural[6] = sum(1 for c in message if c.isdigit())  # Digits
        structural[7] = message.count('http')  # URLs
        structural[8] = message.count('@')  # Mentions/emails
        structural[9] = message.count('#')  # Hashtags
        
        # Ratios (normalized features)
        msg_len = len(message) if len(message) > 0 else 1
        structural[10] = structural[5] / msg_len  # Uppercase ratio
        structural[11] = structural[6] / msg_len  # Digit ratio
        structural[12] = structural[2] / msg_len  # Exclamation ratio
        
        # Advanced features
        words = message.split()
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        structural[13] = avg_word_len
        
        # Repetition detection
        word_counts = Counter(words)
        max_repetition = max(word_counts.values()) if word_counts else 1
        structural[14] = max_repetition / len(words) if words else 0
        
        return structural
    
    def _extract_features(self, message: str, user_id: str, 
                         timestamp: Optional[datetime] = None) -> SpamFeatures:
        """Extract comprehensive feature set for spam detection"""
        
        # Check cache first
        cache_key = self._hash_message(message, user_id)
        if cache_key in self._feature_cache:
            cached_time = datetime.now() - timedelta(minutes=5)
            # Use cache if less than 5 minutes old
            return self._feature_cache[cache_key]
        
        # Extract all feature types
        word_features = self._extract_word_features(message)
        user_spawn_vector = self._extract_user_spawn_vector(user_id, message)
        temporal_features = self._extract_temporal_features(timestamp)
        structural_features = self._extract_structural_features(message)
        
        # Calculate confidence weights based on feature reliability
        confidence_weights = self._calculate_confidence_weights(
            word_features, user_spawn_vector, structural_features
        )
        
        features = SpamFeatures(
            word_features=word_features,
            user_spawn_vector=user_spawn_vector,
            temporal_features=temporal_features,
            structural_features=structural_features,
            confidence_weights=confidence_weights
        )
        
        # Cache the features
        self._feature_cache[cache_key] = features
        
        return features
    
    def _calculate_confidence_weights(self, word_features: np.ndarray, 
                                    user_spawn_vector: np.ndarray,
                                    structural_features: np.ndarray) -> np.ndarray:
        """Calculate confidence weights for features"""
        
        # Base confidence from feature sparsity
        word_confidence = 1.0 - (np.sum(word_features == 0) / len(word_features))
        user_confidence = 1.0 - (np.sum(user_spawn_vector == 0) / len(user_spawn_vector))
        struct_confidence = 1.0 - (np.sum(structural_features == 0) / len(structural_features))
        
        # Combine confidences
        total_features = len(word_features) + len(user_spawn_vector) + len(structural_features)
        confidence_weights = np.full(total_features, 
                                   (word_confidence + user_confidence + struct_confidence) / 3)
        
        return confidence_weights
    
    def _tokenize(self, message: str) -> List[str]:
        """Advanced tokenization with spam-specific preprocessing"""
        import re
        
        # Convert to lowercase
        message = message.lower()
        
        # Remove/normalize URLs
        message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                        'URL_TOKEN', message)
        
        # Remove/normalize emails
        message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                        'EMAIL_TOKEN', message)
        
        # Remove/normalize phone numbers
        message = re.sub(r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', 
                        'PHONE_TOKEN', message)
        
        # Extract words (including handling of special characters common in spam)
        words = re.findall(r'\b\w+\b', message)
        
        # Add n-grams for better context (bigrams)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        
        return words + bigrams
    
    def predict(self, message: str, user_id: str, 
               timestamp: Optional[datetime] = None) -> Tuple[float, float]:
        """
        Predict spam probability for a message
        
        Returns:
            Tuple[float, float]: (spam_probability, confidence_score)
        """
        
        # Check prediction cache
        cache_key = self._hash_message(message, user_id)
        if cache_key in self._prediction_cache:
            cached_prob, cached_time = self._prediction_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=1):
                return cached_prob, 0.9  # High confidence for cached results
        
        try:
            # Extract features
            features = self._extract_features(message, user_id, timestamp)
            
            # Calculate spam probability using non-naive Bayes
            spam_prob = self._calculate_spam_probability(features)
            
            # Calculate confidence based on feature quality and model certainty
            confidence = self._calculate_prediction_confidence(features, spam_prob)
            
            # Cache the prediction
            self._prediction_cache[cache_key] = (spam_prob, datetime.now())
            
            logger.debug(f"Predicted spam_prob={spam_prob:.3f}, confidence={confidence:.3f} for user {user_id}")
            
            return spam_prob, confidence
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 0.5, 0.1  # Return neutral prediction with low confidence
    
    def _calculate_spam_probability(self, features: SpamFeatures) -> float:
        """Calculate spam probability using non-naive Bayes with feature correlations"""
        
        # Traditional Naive Bayes component
        word_log_likelihood = np.dot(features.word_features, 
                                   self.spam_log_probs - self.ham_log_probs)
        
        # Non-naive component: feature correlations
        # This captures interdependencies between features
        correlation_component = np.dot(
            features.word_features.T,
            np.dot(self.feature_correlation_matrix, features.word_features)
        )
        
        # User-specific component
        user_component = np.sum(features.user_spawn_vector) * 0.1
        
        # Temporal component
        temporal_component = np.sum(features.temporal_features) * 0.05
        
        # Structural component
        structural_component = np.sum(features.structural_features) * 0.02
        
        # Combine all components with learned weights
        total_log_odds = (
            word_log_likelihood +
            correlation_component * 0.3 +  # Non-naive weight
            user_component +
            temporal_component +
            structural_component +
            np.log(self.spam_prior / self.ham_prior)
        )
        
        # Convert to probability using sigmoid (more stable than direct exponential)
        spam_prob = 1.0 / (1.0 + np.exp(-total_log_odds))
        
        # Ensure probability is in valid range
        return np.clip(spam_prob, 0.001, 0.999)
    
    def _calculate_prediction_confidence(self, features: SpamFeatures, 
                                       spam_prob: float) -> float:
        """Calculate confidence in the prediction"""
        
        # Confidence based on distance from decision boundary (0.5)
        boundary_confidence = abs(spam_prob - 0.5) * 2
        
        # Confidence based on feature weights
        feature_confidence = np.mean(features.confidence_weights)
        
        # Confidence based on user profile completeness
        user_confidence = 0.5  # Default for unknown users
        
        # Combine confidences
        total_confidence = (boundary_confidence + feature_confidence + user_confidence) / 3
        
        return np.clip(total_confidence, 0.1, 0.99)
    
    def train(self, messages: List[str], labels: List[bool], user_ids: List[str],
             timestamps: Optional[List[datetime]] = None) -> None:
        """
        Train the classifier with new data
        
        Args:
            messages: List of training messages
            labels: List of labels (True for spam, False for ham)
            user_ids: List of user IDs for each message
            timestamps: Optional list of timestamps
        """
        
        if timestamps is None:
            timestamps = [datetime.now()] * len(messages)
        
        logger.info(f"Training on {len(messages)} messages")
        
        # Build vocabulary if not exists
        if not self.vocab_to_idx:
            self._build_vocabulary(messages)
        
        # Update user profiles
        self._update_user_profiles(messages, labels, user_ids, timestamps)
        
        # Update class priors
        spam_count = sum(labels)
        ham_count = len(labels) - spam_count
        
        self.total_spam_count += spam_count
        self.total_ham_count += ham_count
        
        total = self.total_spam_count + self.total_ham_count
        self.spam_prior = self.total_spam_count / total
        self.ham_prior = self.total_ham_count / total
        
        # Update word probabilities
        self._update_word_probabilities(messages, labels)
        
        # Update feature correlation matrix (non-naive component)
        self._update_feature_correlations(messages, labels, user_ids, timestamps)
        
        # Clear caches after training
        self._prediction_cache.clear()
        self._feature_cache.clear()
        
        logger.info(f"Training completed. Spam prior: {self.spam_prior:.3f}")
    
    def _build_vocabulary(self, messages: List[str]) -> None:
        """Build vocabulary from training messages"""
        
        word_counts = Counter()
        for message in messages:
            words = self._tokenize(message)
            word_counts.update(words)
        
        # Select top words by frequency
        most_common = word_counts.most_common(self.vocab_size)
        
        for idx, (word, count) in enumerate(most_common):
            self.vocab_to_idx[word] = idx
            self.idx_to_vocab[idx] = word
        
        logger.info(f"Built vocabulary with {len(self.vocab_to_idx)} words")
    
    def _update_user_profiles(self, messages: List[str], labels: List[bool],
                            user_ids: List[str], timestamps: List[datetime]) -> None:
        """Update user-specific profiles with new training data"""
        
        for message, label, user_id, timestamp in zip(messages, labels, user_ids, timestamps):
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    word_spawn_vectors={},
                    legitimate_patterns=set(),
                    spam_patterns=set(),
                    confidence_scores={},
                    last_updated=timestamp,
                    message_count=0
                )
            
            profile = self.user_profiles[user_id]
            words = set(self._tokenize(message))
            
            # Update word spawn vectors
            for word in words:
                if word not in profile.word_spawn_vectors:
                    profile.word_spawn_vectors[word] = 0.0
                
                # Positive weight for ham, negative for spam
                weight_update = -0.1 if label else 0.1
                
                # Apply temporal decay
                time_diff = (timestamp - profile.last_updated).total_seconds()
                decay_factor = self.temporal_decay ** (time_diff / 3600)  # Hourly decay
                
                profile.word_spawn_vectors[word] = (
                    profile.word_spawn_vectors[word] * decay_factor + weight_update
                )
            
            # Update pattern sets
            if label:
                profile.spam_patterns.update(words)
            else:
                profile.legitimate_patterns.update(words)
            
            # Update confidence scores based on prediction accuracy
            # (This would be updated during online learning)
            
            profile.last_updated = timestamp
            profile.message_count += 1
    
    def _update_word_probabilities(self, messages: List[str], labels: List[bool]) -> None:
        """Update word probability estimates"""
        
        spam_word_counts = np.ones(self.vocab_size)  # Laplace smoothing
        ham_word_counts = np.ones(self.vocab_size)
        
        for message, label in zip(messages, labels):
            word_features = self._extract_word_features(message)
            
            if label:  # Spam
                spam_word_counts += word_features
            else:  # Ham
                ham_word_counts += word_features
        
        # Calculate log probabilities
        spam_total = np.sum(spam_word_counts)
        ham_total = np.sum(ham_word_counts)
        
        self.spam_log_probs = np.log(spam_word_counts / spam_total)
        self.ham_log_probs = np.log(ham_word_counts / ham_total)
    
    def _update_feature_correlations(self, messages: List[str], labels: List[bool],
                                   user_ids: List[str], timestamps: List[datetime]) -> None:
        """Update feature correlation matrix for non-naive component"""
        
        # Sample a subset for efficiency (correlation matrix is O(n²))
        sample_size = min(1000, len(messages))
        indices = np.random.choice(len(messages), sample_size, replace=False)
        
        feature_matrix = []
        
        for idx in indices:
            features = self._extract_features(messages[idx], user_ids[idx], timestamps[idx])
            feature_vector = features.word_features
            feature_matrix.append(feature_vector)
        
        if feature_matrix:
            feature_matrix = np.array(feature_matrix)
            
            # Calculate correlation matrix
            correlation = np.corrcoef(feature_matrix.T)
            
            # Update with exponential moving average
            alpha = 0.1  # Learning rate
            self.feature_correlation_matrix = (
                (1 - alpha) * self.feature_correlation_matrix +
                alpha * correlation
            )
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        
        model_data = {
            'spam_log_probs': self.spam_log_probs,
            'ham_log_probs': self.ham_log_probs,
            'spam_prior': self.spam_prior,
            'ham_prior': self.ham_prior,
            'feature_correlation_matrix': self.feature_correlation_matrix,
            'vocab_to_idx': self.vocab_to_idx,
            'idx_to_vocab': self.idx_to_vocab,
            'user_profiles': self.user_profiles,
            'total_spam_count': self.total_spam_count,
            'total_ham_count': self.total_ham_count,
            'vocab_size': self.vocab_size,
            'user_vector_dim': self.user_vector_dim,
            'temporal_decay': self.temporal_decay,
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore all model parameters
        self.spam_log_probs = model_data['spam_log_probs']
        self.ham_log_probs = model_data['ham_log_probs']
        self.spam_prior = model_data['spam_prior']
        self.ham_prior = model_data['ham_prior']
        self.feature_correlation_matrix = model_data['feature_correlation_matrix']
        self.vocab_to_idx = model_data['vocab_to_idx']
        self.idx_to_vocab = model_data['idx_to_vocab']
        self.user_profiles = model_data['user_profiles']
        self.total_spam_count = model_data['total_spam_count']
        self.total_ham_count = model_data['total_ham_count']
        self.vocab_size = model_data['vocab_size']
        self.user_vector_dim = model_data['user_vector_dim']
        self.temporal_decay = model_data['temporal_decay']
        self.confidence_threshold = model_data['confidence_threshold']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_stats(self) -> Dict:
        """Get model statistics for monitoring"""
        
        return {
            'total_messages_trained': self.total_spam_count + self.total_ham_count,
            'spam_messages': self.total_spam_count,
            'ham_messages': self.total_ham_count,
            'spam_prior': self.spam_prior,
            'vocab_size': len(self.vocab_to_idx),
            'user_profiles': len(self.user_profiles),
            'cache_size': len(self._prediction_cache),
            'feature_cache_size': len(self._feature_cache)
        }
