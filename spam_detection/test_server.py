#!/usr/bin/env python3
"""
Minimal test server for spam detection API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import time
import re
import hashlib
from datetime import datetime

app = FastAPI(title="Spam Detection Test Server", version="1.0.0")

# Simple in-memory storage for testing
spam_stats = {
    "total_messages_processed": 0,
    "spam_detected": 0,
    "ham_detected": 0,
    "training_samples": 0
}

# Basic spam patterns for testing
SPAM_PATTERNS = [
    r'(?i)(viagra|cialis|pharmacy)',
    r'(?i)(win|won|winner).*(money|cash|prize)',
    r'(?i)(click|visit).*(link|website)',
    r'(?i)(free|cheap).*(offer|deal)',
    r'(?i)(urgent|act now|limited time)',
    r'(?i)(bitcoin|crypto|investment)',
    r'\$\d+',  # Money amounts
    r'(?i)(loan|debt|credit)',
    r'(?i)(weight loss|diet pill)',
    r'(?i)(nigerian prince|inheritance)'
]

class MessageRequest(BaseModel):
    message: str
    user_id: str
    timestamp: str = None

class BatchRequest(BaseModel):
    messages: List[Dict[str, Any]]

class TrainingRequest(BaseModel):
    message: str
    user_id: str
    is_spam: bool
    timestamp: str = None

def simple_spam_detection(message: str, user_id: str) -> Dict[str, Any]:
    """Simple rule-based spam detection for testing"""
    
    # Count spam patterns
    spam_score = 0
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, message):
            spam_score += 1
    
    # Simple heuristics
    word_count = len(message.split())
    if word_count < 3:
        spam_score += 0.5
    
    # Check for excessive capitalization
    if len(re.findall(r'[A-Z]', message)) > len(message) * 0.5:
        spam_score += 1
    
    # Check for excessive punctuation
    if len(re.findall(r'[!?]{2,}', message)) > 0:
        spam_score += 0.5
    
    # Normalize score
    max_possible_score = len(SPAM_PATTERNS) + 3
    spam_probability = min(spam_score / max_possible_score, 1.0)
    
    # Determine if spam - lowered threshold for better detection
    is_spam = spam_probability > 0.15  # Lowered from 0.4 to 0.15
    confidence = spam_probability if is_spam else (1 - spam_probability)
    
    return {
        "is_spam": is_spam,
        "spam_probability": round(spam_probability, 3),
        "confidence": round(confidence, 3),
        "processing_time_ms": round(time.time() * 1000 % 1000, 2),
        "patterns_matched": int(spam_score)
    }

@app.get("/")
async def root():
    return {"message": "Spam Detection Test Server", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict_spam(request: MessageRequest):
    try:
        result = simple_spam_detection(request.message, request.user_id)
        
        # Update stats
        spam_stats["total_messages_processed"] += 1
        if result["is_spam"]:
            spam_stats["spam_detected"] += 1
        else:
            spam_stats["ham_detected"] += 1
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_spam_batch(request: BatchRequest):
    try:
        results = []
        for msg_data in request.messages:
            message = msg_data.get("message", "")
            user_id = msg_data.get("user_id", "unknown")
            result = simple_spam_detection(message, user_id)
            results.append(result)
            
            # Update stats
            spam_stats["total_messages_processed"] += 1
            if result["is_spam"]:
                spam_stats["spam_detected"] += 1
            else:
                spam_stats["ham_detected"] += 1
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/train")
async def submit_training_data(request: TrainingRequest):
    try:
        # In a real implementation, this would update the model
        # For now, just update stats
        spam_stats["training_samples"] += 1
        
        return {
            "status": "success", 
            "message": "Training data received",
            "is_spam": request.is_spam,
            "user_id": request.user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/train/batch")
async def submit_training_data_batch(request: List[TrainingRequest]):
    try:
        spam_stats["training_samples"] += len(request)
        return {
            "status": "success", 
            "message": f"Received {len(request)} training samples",
            "samples_processed": len(request)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch training failed: {str(e)}")

@app.get("/stats")
async def get_model_stats():
    return {
        "model_stats": spam_stats,
        "server_info": {
            "version": "1.0.0-test",
            "type": "rule-based",
            "patterns_count": len(SPAM_PATTERNS)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    total = spam_stats["total_messages_processed"]
    if total == 0:
        return {"accuracy": 0, "spam_rate": 0, "ham_rate": 0}
    
    return {
        "total_processed": total,
        "spam_rate": round(spam_stats["spam_detected"] / total, 3),
        "ham_rate": round(spam_stats["ham_detected"] / total, 3),
        "training_samples": spam_stats["training_samples"]
    }

if __name__ == "__main__":
    print("Starting Spam Detection Test Server...")
    print("Server will be available at: http://localhost:8082")
    print("API Documentation at: http://localhost:8082/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8082,
        log_level="info"
    )
