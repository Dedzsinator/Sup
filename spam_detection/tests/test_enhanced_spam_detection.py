"""
Tests for enhanced spam detection system
"""

import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from enhanced_trainer import EnhancedSpamTrainer
    from advanced_classifier import AdvancedSpamClassifier
    from enhanced_preprocessor import EnhancedPreprocessor
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

from server import app

# Test client for API testing
client = TestClient(app)

class TestSpamDetectionAPI:
    """Test the spam detection API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_type" in data
        assert "enhanced_available" in data
    
    def test_single_prediction(self):
        """Test single message prediction"""
        test_message = {
            "text": "Buy cheap viagra now! Click here to save money!",
            "user_id": "test_user"
        }
        
        response = client.post("/predict", json=test_message)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_spam" in data
        assert "confidence" in data
        assert "model_type" in data
        assert "processing_time_ms" in data
        assert isinstance(data["is_spam"], bool)
        assert 0 <= data["confidence"] <= 1
    
    def test_batch_prediction(self):
        """Test batch message prediction"""
        test_batch = {
            "messages": [
                {"text": "Buy cheap viagra now!", "user_id": "test_user1"},
                {"text": "Meeting tomorrow at 3pm", "user_id": "test_user2"},
                {"text": "URGENT: You've won $1000000!", "user_id": "test_user3"}
            ]
        }
        
        response = client.post("/predict/batch", json=test_batch)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_processed" in data
        assert "total_spam" in data
        assert "total_ham" in data
        assert "average_confidence" in data
        assert len(data["predictions"]) == 3
        assert data["total_processed"] == 3
    
    def test_statistics(self):
        """Test statistics endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_messages_processed" in data
        assert "spam_detected" in data
        assert "ham_detected" in data
        assert "model_type" in data
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "enhanced_available" in data
        assert "version" in data
    
    def test_feedback_submission(self):
        """Test feedback submission"""
        response = client.post(
            "/feedback",
            params={
                "message_text": "This is a test message",
                "is_spam": True,
                "user_id": "test_user"
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "feedback_received"

@pytest.mark.skipif(not ENHANCED_AVAILABLE, reason="Enhanced components not available")
class TestEnhancedComponents:
    """Test enhanced ML components if available"""
    
    def test_enhanced_preprocessor_import(self):
        """Test that enhanced preprocessor can be imported"""
        assert EnhancedPreprocessor is not None
    
    def test_advanced_classifier_import(self):
        """Test that advanced classifier can be imported"""
        assert AdvancedSpamClassifier is not None
    
    def test_enhanced_trainer_import(self):
        """Test that enhanced trainer can be imported"""
        assert EnhancedSpamTrainer is not None

class TestRuleBasedDetection:
    """Test rule-based spam detection fallback"""
    
    def test_spam_patterns(self):
        """Test basic spam pattern detection"""
        # Import the rule-based function from server
        from server import rule_based_classify
        
        spam_messages = [
            "Buy cheap viagra now!",
            "You've won $1000000!",
            "URGENT: Click here now!",
            "Free loan approval!"
        ]
        
        ham_messages = [
            "Meeting tomorrow at 3pm",
            "Thanks for dinner",
            "How are you doing?",
            "See you later"
        ]
        
        # Test spam detection
        for message in spam_messages:
            result = rule_based_classify(message)
            assert result["is_spam"] == True, f"Failed to detect spam: {message}"
            assert result["confidence"] > 0
        
        # Test ham detection (most should be classified as ham)
        ham_correct = 0
        for message in ham_messages:
            result = rule_based_classify(message)
            if not result["is_spam"]:
                ham_correct += 1
        
        # At least 75% of ham messages should be correctly classified
        assert ham_correct >= len(ham_messages) * 0.75
    
    def test_excessive_capitals(self):
        """Test excessive capitals detection"""
        from server import rule_based_classify
        
        result = rule_based_classify("THIS IS ALL CAPS MESSAGE!!!")
        assert result["is_spam"] == True
        assert "excessive_capitals" in result["matched_patterns"]
    
    def test_excessive_punctuation(self):
        """Test excessive punctuation detection"""
        from server import rule_based_classify
        
        result = rule_based_classify("Buy now!!! Amazing deal???")
        assert "excessive_punctuation" in result["matched_patterns"]

class TestDataValidation:
    """Test input validation and edge cases"""
    
    def test_empty_message(self):
        """Test handling of empty messages"""
        response = client.post("/predict", json={"text": "", "user_id": "test"})
        assert response.status_code == 200  # Should handle gracefully
    
    def test_very_long_message(self):
        """Test handling of very long messages"""
        long_text = "A" * 10000  # 10k characters
        response = client.post("/predict", json={"text": long_text, "user_id": "test"})
        assert response.status_code == 200
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = "Message with émojis 🚀 and spëcial chars ñ"
        response = client.post("/predict", json={"text": special_text, "user_id": "test"})
        assert response.status_code == 200
    
    def test_invalid_json(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
