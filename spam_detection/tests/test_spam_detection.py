"""
Tests for hyperoptimized spam detection system
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
import os
from unittest.mock import Mock, patch

# Import modules to test
import sys
sys.path.append('/app/src')

from classifier import HyperoptimizedBayesClassifier, SpamFeatures, UserProfile
from trainer import SpamDetectionTrainer
from utils import TextPreprocessor, FeatureEngineering, DataValidator
from server import app

class TestHyperoptimizedBayesClassifier:
    """Test suite for the spam classifier"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = HyperoptimizedBayesClassifier(
            vocab_size=1000,
            user_vector_dim=64
        )
        
        # Sample training data
        self.spam_messages = [
            "URGENT! Win $1000 now! Click here: http://spam.com",
            "Free iPhone! Limited time offer! Text 555-SPAM now!",
            "Make money fast! Work from home! email@spam.com",
            "Your account will be suspended! Click http://fake.com immediately!"
        ]
        
        self.ham_messages = [
            "Hey, how are you doing today?",
            "Can we meet for lunch tomorrow?",
            "Thanks for the help with the project!",
            "The meeting is scheduled for 2 PM"
        ]
        
        self.user_ids = ['user1', 'user2', 'user1', 'user3', 'user1', 'user2', 'user3', 'user1']
        self.labels = [True, True, True, True, False, False, False, False]
        self.messages = self.spam_messages + self.ham_messages
        self.timestamps = [datetime.now() - timedelta(days=i) for i in range(8)]
    
    def test_initialization(self):
        """Test classifier initialization"""
        assert self.classifier.vocab_size == 1000
        assert self.classifier.user_vector_dim == 64
        assert self.classifier.temporal_decay == 0.95
        assert len(self.classifier.user_profiles) == 0
    
    def test_tokenization(self):
        """Test message tokenization"""
        message = "URGENT! Win $1000 now! Click here: http://spam.com"
        tokens = self.classifier._tokenize(message)
        
        assert 'urgent' in tokens
        assert 'win' in tokens
        assert 'URL_TOKEN' in tokens
        assert len(tokens) > 0
    
    def test_feature_extraction(self):
        """Test feature extraction"""
        message = "Free money! Click now!"
        user_id = "test_user"
        
        features = self.classifier._extract_features(message, user_id)
        
        assert isinstance(features, SpamFeatures)
        assert len(features.word_features) == self.classifier.vocab_size
        assert len(features.user_spawn_vector) == self.classifier.user_vector_dim
        assert len(features.temporal_features) == 10
        assert len(features.structural_features) == 15
    
    def test_training(self):
        """Test model training"""
        self.classifier.train(self.messages, self.labels, self.user_ids, self.timestamps)
        
        # Check that vocabulary was built
        assert len(self.classifier.vocab_to_idx) > 0
        
        # Check that user profiles were created
        assert len(self.classifier.user_profiles) > 0
        
        # Check that priors were updated
        assert 0 < self.classifier.spam_prior < 1
        assert 0 < self.classifier.ham_prior < 1
        assert abs(self.classifier.spam_prior + self.classifier.ham_prior - 1.0) < 1e-6
    
    def test_prediction(self):
        """Test spam prediction"""
        # Train first
        self.classifier.train(self.messages, self.labels, self.user_ids, self.timestamps)
        
        # Test spam message
        spam_prob, confidence = self.classifier.predict("FREE MONEY! WIN NOW! CLICK HERE!", "test_user")
        assert 0 <= spam_prob <= 1
        assert 0 <= confidence <= 1
        
        # Test ham message
        ham_prob, ham_confidence = self.classifier.predict("How are you doing today?", "test_user")
        assert 0 <= ham_prob <= 1
        assert 0 <= ham_confidence <= 1
    
    def test_user_profile_updates(self):
        """Test user profile updates during training"""
        self.classifier.train(self.messages, self.labels, self.user_ids, self.timestamps)
        
        # Check that user profiles exist
        assert 'user1' in self.classifier.user_profiles
        assert 'user2' in self.classifier.user_profiles
        
        # Check profile structure
        profile = self.classifier.user_profiles['user1']
        assert isinstance(profile, UserProfile)
        assert profile.user_id == 'user1'
        assert len(profile.word_spawn_vectors) > 0
        assert profile.message_count > 0
    
    def test_model_persistence(self):
        """Test model save/load functionality"""
        # Train the model
        self.classifier.train(self.messages, self.labels, self.user_ids, self.timestamps)
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            self.classifier.save_model(model_path)
            
            # Create new classifier and load model
            new_classifier = HyperoptimizedBayesClassifier()
            new_classifier.load_model(model_path)
            
            # Test that loaded model works
            original_pred = self.classifier.predict("Test message", "test_user")
            loaded_pred = new_classifier.predict("Test message", "test_user")
            
            # Predictions should be very similar
            assert abs(original_pred[0] - loaded_pred[0]) < 0.01
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_caching(self):
        """Test prediction caching"""
        self.classifier.train(self.messages, self.labels, self.user_ids, self.timestamps)
        
        message = "Test message for caching"
        user_id = "cache_test_user"
        
        # First prediction
        pred1 = self.classifier.predict(message, user_id)
        
        # Second prediction (should use cache)
        pred2 = self.classifier.predict(message, user_id)
        
        # Should be identical
        assert pred1[0] == pred2[0]
        assert pred1[1] == pred2[1]

class TestSpamDetectionTrainer:
    """Test suite for the training pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.trainer = SpamDetectionTrainer()
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        messages, labels, user_ids, timestamps = self.trainer._generate_synthetic_data(n_samples=100)
        
        assert len(messages) == 100
        assert len(labels) == 100
        assert len(user_ids) == 100
        assert len(timestamps) == 100
        
        # Check label distribution (should be roughly 30% spam, 70% ham)
        spam_count = sum(labels)
        assert 20 <= spam_count <= 40  # Allow some variance
    
    def test_model_training(self):
        """Test complete training pipeline"""
        messages, labels, user_ids, timestamps = self.trainer._generate_synthetic_data(n_samples=50)
        
        results = self.trainer.train_model(messages, labels, user_ids, timestamps)
        
        assert 'test_metrics' in results
        assert 'cv_metrics' in results
        assert 'model_stats' in results
        
        # Check that metrics are reasonable
        assert 0 <= results['test_metrics']['accuracy'] <= 1
        assert 0 <= results['test_metrics']['f1'] <= 1

class TestTextPreprocessor:
    """Test suite for text preprocessing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = TextPreprocessor()
    
    def test_text_normalization(self):
        """Test text normalization"""
        text = "  HELLO    WORLD!  "
        normalized = self.preprocessor.normalize_text(text)
        
        assert normalized == "hello world!"
    
    def test_spam_feature_extraction(self):
        """Test spam feature extraction"""
        spam_text = "URGENT! Win $1000 NOW! Call 555-123-4567!"
        features = self.preprocessor.extract_spam_features(spam_text)
        
        assert 'urgency_count' in features
        assert 'money_count' in features
        assert 'contact_info_count' in features
        assert features['urgency_count'] > 0
        assert features['money_count'] > 0
        assert features['contact_info_count'] > 0
    
    def test_text_cleaning(self):
        """Test text cleaning"""
        dirty_text = "Check out http://spam.com or email me@spam.com or call 555-123-4567!!!"
        clean_text = self.preprocessor.clean_text(dirty_text)
        
        assert '[URL]' in clean_text
        assert '[EMAIL]' in clean_text
        assert '[PHONE]' in clean_text
        assert '!!!' not in clean_text  # Excessive punctuation removed

class TestFeatureEngineering:
    """Test suite for feature engineering"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.feature_eng = FeatureEngineering()
    
    def test_ngram_extraction(self):
        """Test n-gram extraction"""
        text = "hello world test"
        bigrams = self.feature_eng.extract_ngrams(text, n=2)
        
        assert 'hello_world' in bigrams
        assert 'world_test' in bigrams
        assert len(bigrams) == 2
    
    def test_character_ngrams(self):
        """Test character n-gram extraction"""
        text = "hello"
        trigrams = self.feature_eng.extract_character_ngrams(text, n=3)
        
        assert 'hel' in trigrams
        assert 'ell' in trigrams
        assert 'llo' in trigrams
        assert len(trigrams) == 3
    
    def test_syntactic_features(self):
        """Test syntactic feature extraction"""
        text = "What is this? Click here now! Amazing offer!!!"
        features = self.feature_eng.extract_syntactic_features(text)
        
        assert 'question_words' in features
        assert 'action_words' in features
        assert 'emotion_words' in features
        assert 'exclamation_count' in features
        
        assert features['question_words'] > 0
        assert features['action_words'] > 0
        assert features['emotion_words'] > 0

class TestDataValidator:
    """Test suite for data validation"""
    
    def test_message_validation(self):
        """Test message validation"""
        # Valid messages
        assert DataValidator.validate_message("Hello world")
        assert DataValidator.validate_message("This is a test message")
        
        # Invalid messages
        assert not DataValidator.validate_message("")
        assert not DataValidator.validate_message("   ")
        assert not DataValidator.validate_message("Hi")  # Too short
        assert not DataValidator.validate_message("!" * 10001)  # Too long
        assert not DataValidator.validate_message("!@#$%")  # Only special chars
    
    def test_user_id_validation(self):
        """Test user ID validation"""
        # Valid user IDs
        assert DataValidator.validate_user_id("user123")
        assert DataValidator.validate_user_id("test_user")
        assert DataValidator.validate_user_id("user-123")
        
        # Invalid user IDs
        assert not DataValidator.validate_user_id("")
        assert not DataValidator.validate_user_id("   ")
        assert not DataValidator.validate_user_id("user@domain.com")  # Invalid chars
        assert not DataValidator.validate_user_id("a" * 101)  # Too long
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        dirty_input = "Hello\x00World   \n\n\n   "
        clean_input = DataValidator.sanitize_input(dirty_input)
        
        assert '\x00' not in clean_input
        assert clean_input == "Hello World"

# Integration tests for the API
class TestAPI:
    """Test suite for the API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_prediction_endpoint_without_auth(self, client):
        """Test prediction endpoint without authentication"""
        response = client.post("/predict", json={
            "message": "Test message",
            "user_id": "test_user"
        })
        # Should require authentication
        assert response.status_code == 401
    
    def test_prediction_endpoint_with_auth(self, client):
        """Test prediction endpoint with authentication"""
        headers = {"Authorization": "Bearer test-token"}
        response = client.post("/predict", json={
            "message": "Test message",
            "user_id": "test_user"
        }, headers=headers)
        
        # May return 503 if no model is loaded, but should not be 401
        assert response.status_code != 401

if __name__ == "__main__":
    pytest.main(["-v", __file__])
