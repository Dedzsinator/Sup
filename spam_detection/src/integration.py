"""
Integration module for connecting spam detection with the main Sup backend

This module provides the interface for the Sup backend to communicate with
the spam detection microservice.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class SpamDetectionClient:
    """
    Client for communicating with the spam detection microservice
    """
    
    def __init__(self, 
                 base_url: str = None,
                 api_key: str = None,
                 timeout: int = 5):
        
        self.base_url = base_url or os.getenv("SPAM_DETECTION_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("SPAM_DETECTION_API_KEY", "your-secret-api-key")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
        
        logger.info(f"Initialized SpamDetectionClient with base_url={self.base_url}")
    
    async def check_spam(self, 
                        message: str, 
                        user_id: str, 
                        timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Check if a message is spam
        
        Returns:
            Dict with keys: is_spam, spam_probability, confidence, processing_time_ms
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        payload = {
            "message": message,
            "user_id": user_id,
            "timestamp": timestamp.isoformat()
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.debug(f"Spam check result: {result}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Spam detection API error {response.status}: {error_text}")
                        return self._get_fallback_result()
                        
        except asyncio.TimeoutError:
            logger.warning("Spam detection request timed out")
            return self._get_fallback_result()
        except Exception as e:
            logger.error(f"Spam detection request failed: {e}")
            return self._get_fallback_result()
    
    async def check_spam_batch(self, 
                              messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check multiple messages for spam
        
        Args:
            messages: List of dicts with keys: message, user_id, timestamp (optional)
        
        Returns:
            List of spam check results
        """
        
        # Prepare batch payload
        batch_messages = []
        for msg_data in messages:
            payload_item = {
                "message": msg_data["message"],
                "user_id": msg_data["user_id"]
            }
            
            if "timestamp" in msg_data:
                if isinstance(msg_data["timestamp"], datetime):
                    payload_item["timestamp"] = msg_data["timestamp"].isoformat()
                else:
                    payload_item["timestamp"] = msg_data["timestamp"]
            
            batch_messages.append(payload_item)
        
        payload = {"messages": batch_messages}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/predict/batch",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        results = await response.json()
                        logger.debug(f"Batch spam check completed: {len(results)} results")
                        return results
                    else:
                        error_text = await response.text()
                        logger.error(f"Batch spam detection API error {response.status}: {error_text}")
                        return [self._get_fallback_result() for _ in messages]
                        
        except asyncio.TimeoutError:
            logger.warning("Batch spam detection request timed out")
            return [self._get_fallback_result() for _ in messages]
        except Exception as e:
            logger.error(f"Batch spam detection request failed: {e}")
            return [self._get_fallback_result() for _ in messages]
    
    async def submit_training_data(self, 
                                 message: str, 
                                 user_id: str, 
                                 is_spam: bool,
                                 timestamp: Optional[datetime] = None) -> bool:
        """
        Submit training data to improve the model
        
        Returns:
            True if successful, False otherwise
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        payload = {
            "message": message,
            "user_id": user_id,
            "is_spam": is_spam,
            "timestamp": timestamp.isoformat()
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/train",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        logger.debug("Training data submitted successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Training data submission failed {response.status}: {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Training data submission failed: {e}")
            return False
    
    async def get_model_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get model statistics
        
        Returns:
            Dict with model statistics or None if failed
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(
                    f"{self.base_url}/stats",
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        stats = await response.json()
                        return stats
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to get model stats {response.status}: {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"Failed to get model stats: {e}")
            return None
    
    async def health_check(self) -> bool:
        """
        Check if the spam detection service is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _get_fallback_result(self) -> Dict[str, Any]:
        """
        Get fallback result when spam detection service is unavailable
        """
        return {
            "is_spam": False,  # Default to not spam to avoid blocking messages
            "spam_probability": 0.5,
            "confidence": 0.1,
            "processing_time_ms": 0.0,
            "user_id": "unknown",
            "error": "Spam detection service unavailable"
        }

# Global client instance
_spam_client = None

def get_spam_client() -> SpamDetectionClient:
    """Get global spam detection client instance"""
    global _spam_client
    if _spam_client is None:
        _spam_client = SpamDetectionClient()
    return _spam_client

async def check_message_spam(message: str, user_id: str) -> Dict[str, Any]:
    """
    Convenience function to check if a message is spam
    """
    client = get_spam_client()
    return await client.check_spam(message, user_id)

async def report_spam(message: str, user_id: str, is_spam: bool) -> bool:
    """
    Convenience function to report spam/ham for training
    """
    client = get_spam_client()
    return await client.submit_training_data(message, user_id, is_spam)
