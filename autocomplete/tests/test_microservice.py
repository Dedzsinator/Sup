#!/usr/bin/env python3
"""
Test suite for the autocomplete microservice.

This module tests the microservice API endpoints and integration with the Elixir backend.
"""

import pytest
import asyncio
import json
import time
import requests
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the microservice app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api_server_microservice import app

# Test client
client = TestClient(app)

class TestMicroserviceAPI:
    """Test cases for the microservice API endpoints."""
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "pipeline_loaded" in data
        assert isinstance(data["uptime_seconds"], (int, float))
    
    def test_root_endpoint(self):
        """Test the root information endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Autocomplete Microservice"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "suggestions" in data["endpoints"]
        assert "completion" in data["endpoints"]
    
    def test_stats_endpoint(self):
        """Test the statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "requests_processed" in data
        assert "average_latency_ms" in data
        assert "trie_size" in data
        assert "model_status" in data
        assert isinstance(data["requests_processed"], int)
        assert isinstance(data["average_latency_ms"], (int, float))
    
    @patch('api_server_microservice.pipeline')
    def test_suggest_endpoint_success(self, mock_pipeline):
        """Test successful suggestion request."""
        # Mock pipeline
        mock_pipeline.get_suggestions.return_value = ["hello world", "hello there"]
        mock_pipeline.__bool__ = lambda self: True
        
        response = client.post("/suggest", json={
            "text": "hello",
            "limit": 5
        })
        
        assert response.status_code == 200
        suggestions = response.json()
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
    
    @patch('api_server_microservice.pipeline')
    def test_suggest_endpoint_with_context(self, mock_pipeline):
        """Test suggestion request with user and room context."""
        mock_pipeline.get_suggestions.return_value = ["contextual suggestion"]
        mock_pipeline.__bool__ = lambda self: True
        
        response = client.post("/suggest", json={
            "text": "test",
            "user_id": "user123",
            "room_id": "room456",
            "limit": 3
        })
        
        assert response.status_code == 200
        suggestions = response.json()
        assert isinstance(suggestions, list)
    
    def test_suggest_endpoint_empty_text(self):
        """Test suggestion request with empty text."""
        response = client.post("/suggest", json={
            "text": "",
            "limit": 5
        })
        
        assert response.status_code == 200
        suggestions = response.json()
        assert suggestions == []
    
    def test_suggest_endpoint_text_too_long(self):
        """Test suggestion request with text that's too long."""
        long_text = "a" * 501  # Exceeds 500 character limit
        
        response = client.post("/suggest", json={
            "text": long_text,
            "limit": 5
        })
        
        assert response.status_code == 400
        assert "Text too long" in response.json()["detail"]
    
    def test_suggest_endpoint_limit_too_high(self):
        """Test suggestion request with limit too high."""
        response = client.post("/suggest", json={
            "text": "test",
            "limit": 25  # Exceeds max limit of 20
        })
        
        assert response.status_code == 400
        assert "Limit too high" in response.json()["detail"]
    
    def test_suggest_endpoint_no_pipeline(self):
        """Test suggestion request when pipeline is not loaded."""
        with patch('api_server_microservice.pipeline', None):
            response = client.post("/suggest", json={
                "text": "test",
                "limit": 5
            })
            
            assert response.status_code == 503
            assert "Pipeline not initialized" in response.json()["detail"]
    
    @patch('api_server_microservice.pipeline')
    def test_complete_endpoint_success(self, mock_pipeline):
        """Test successful completion request."""
        mock_pipeline.generate_completion.return_value = "completed text here"
        mock_pipeline.__bool__ = lambda self: True
        
        response = client.post("/complete", json={
            "text": "I think we should",
            "max_length": 50
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "completion" in data
        assert isinstance(data["completion"], str)
    
    @patch('api_server_microservice.pipeline')
    def test_complete_endpoint_with_context(self, mock_pipeline):
        """Test completion request with context."""
        mock_pipeline.generate_completion.return_value = "contextual completion"
        mock_pipeline.__bool__ = lambda self: True
        
        response = client.post("/complete", json={
            "text": "Hello",
            "user_id": "user123",
            "room_id": "room456",
            "max_length": 30
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "completion" in data
    
    def test_complete_endpoint_empty_text(self):
        """Test completion request with empty text."""
        response = client.post("/complete", json={
            "text": "",
            "max_length": 50
        })
        
        assert response.status_code == 400
        assert "Text is required" in response.json()["detail"]
    
    @patch('api_server_microservice.pipeline')
    def test_train_endpoint_success(self, mock_pipeline):
        """Test successful training request."""
        mock_pipeline.train_incremental = Mock()
        mock_pipeline.__bool__ = lambda self: True
        
        response = client.post("/train", json={
            "messages": [
                {
                    "text": "Hello world",
                    "user_id": "user1",
                    "room_id": "room1",
                    "timestamp": "2025-06-19T10:00:00Z"
                },
                {
                    "text": "How are you?",
                    "user_id": "user2", 
                    "room_id": "room1"
                }
            ]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["messages_count"] == 2

class TestElixirBackendCompatibility:
    """Test compatibility with Elixir backend expectations."""
    
    @patch('api_server_microservice.pipeline')
    def test_elixir_suggest_format(self, mock_pipeline):
        """Test that suggestion response format matches Elixir expectations."""
        mock_pipeline.get_suggestions.return_value = ["suggestion1", "suggestion2"]
        mock_pipeline.__bool__ = lambda self: True
        
        # Test the format expected by Elixir backend
        response = client.post("/suggest", json={
            "text": "hello",
            "max_suggestions": 3  # Elixir uses max_suggestions parameter
        })
        
        assert response.status_code == 200
        suggestions = response.json()
        
        # Should return simple list of strings
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
    
    @patch('api_server_microservice.pipeline')
    def test_elixir_complete_format(self, mock_pipeline):
        """Test that completion response format matches Elixir expectations."""
        mock_pipeline.generate_completion.return_value = "completed text"
        mock_pipeline.__bool__ = lambda self: True
        
        response = client.post("/complete", json={
            "text": "Hello",
            "max_length": 50
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return object with 'completion' key
        assert isinstance(data, dict)
        assert "completion" in data
        assert isinstance(data["completion"], str)

class TestPerformanceAndMetrics:
    """Test performance characteristics and metrics."""
    
    @patch('api_server_microservice.pipeline')
    def test_request_metrics_tracking(self, mock_pipeline):
        """Test that request metrics are properly tracked."""
        mock_pipeline.get_suggestions.return_value = ["test"]
        mock_pipeline.__bool__ = lambda self: True
        
        # Get initial stats
        initial_stats = client.get("/stats").json()
        initial_count = initial_stats["requests_processed"]
        
        # Make a request
        client.post("/suggest", json={"text": "test", "limit": 1})
        
        # Check stats updated
        new_stats = client.get("/stats").json()
        assert new_stats["requests_processed"] == initial_count + 1
        assert new_stats["average_latency_ms"] >= 0
    
    @patch('api_server_microservice.pipeline')
    def test_concurrent_requests(self, mock_pipeline):
        """Test handling of concurrent requests."""
        mock_pipeline.get_suggestions.return_value = ["concurrent"]
        mock_pipeline.__bool__ = lambda self: True
        
        # Simulate concurrent requests
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            response = client.post("/suggest", json={"text": "concurrent", "limit": 1})
            results.put(response.status_code)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        while not results.empty():
            assert results.get() == 200

if __name__ == "__main__":
    # Run basic integration test
    print("🧪 Running Microservice Integration Tests")
    print("=" * 50)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Pipeline loaded: {health_data['pipeline_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            
        # Test suggestion endpoint
        response = requests.post("http://localhost:8000/suggest", 
                               json={"text": "hello", "limit": 3}, 
                               timeout=5)
        if response.status_code == 200:
            suggestions = response.json()
            print(f"✅ Suggestions endpoint passed: {len(suggestions)} suggestions")
        else:
            print(f"❌ Suggestions endpoint failed: {response.status_code}")
            
        print("\n🎉 Integration tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to microservice at http://localhost:8000")
        print("   Make sure the service is running with: ./deploy.sh dev")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")