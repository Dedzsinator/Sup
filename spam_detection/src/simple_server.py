"""
Simple and robust spam detection service for demonstration

This is a simplified version that focuses on functionality over advanced ML
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
from datetime import datetime
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simple Spam Detection API",
    description="Simple spam detection microservice",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple spam patterns
SPAM_PATTERNS = [
    r'\b(urgent|asap|immediate|now|hurry|quick|fast)\b',
    r'\b(free|money|cash|prize|win|won|winner)\b',
    r'\b(click here|click now|limited time|act now)\b',
    r'\$\d+',
    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
    r'http[s]?://\S+',  # URLs
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
]

COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in SPAM_PATTERNS]

# Request/Response models
class SpamCheckRequest(BaseModel):
    message: str
    user_id: str
    timestamp: Optional[datetime] = None

class SpamCheckResponse(BaseModel):
    is_spam: bool
    spam_probability: float
    confidence: float
    processing_time_ms: float
    user_id: str

class BatchSpamCheckRequest(BaseModel):
    messages: List[SpamCheckRequest]

class TrainingDataRequest(BaseModel):
    message: str
    user_id: str
    is_spam: bool
    timestamp: Optional[datetime] = None

class ModelStatsResponse(BaseModel):
    total_messages_trained: int
    spam_messages: int
    ham_messages: int
    uptime_seconds: float

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    model_loaded: bool

# Global statistics
start_time = time.time()
total_checks = 0
spam_detected = 0
training_data_count = 0

def simple_spam_detection(message: str) -> tuple[float, float]:
    """
    Simple pattern-based spam detection
    Returns (spam_probability, confidence)
    """
    if not message:
        return 0.0, 1.0
    
    message_lower = message.lower()
    matches = 0
    
    # Check for spam patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(message):
            matches += 1
    
    # Check for excessive caps
    if len(message) > 10:
        caps_ratio = sum(1 for c in message if c.isupper()) / len(message)
        if caps_ratio > 0.7:
            matches += 1
    
    # Check for excessive punctuation
    punct_count = sum(1 for c in message if c in '!?')
    if punct_count > 3:
        matches += 1
    
    # Calculate spam probability based on matches
    max_possible_matches = len(COMPILED_PATTERNS) + 2  # patterns + caps + punct
    spam_probability = min(matches / max_possible_matches, 0.95)
    
    # Simple confidence calculation
    confidence = 0.8 if matches > 0 else 0.6
    
    return spam_probability, confidence

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        model_loaded=True
    )

@app.post("/predict", response_model=SpamCheckResponse)
async def predict_spam(request: SpamCheckRequest):
    """Predict if a message is spam"""
    global total_checks, spam_detected
    
    start_time_ms = time.time()
    total_checks += 1
    
    try:
        spam_prob, confidence = simple_spam_detection(request.message)
        is_spam = spam_prob > 0.5
        
        if is_spam:
            spam_detected += 1
        
        processing_time = (time.time() - start_time_ms) * 1000
        
        logger.info(f"Spam check: user={request.user_id}, is_spam={is_spam}, prob={spam_prob:.3f}")
        
        return SpamCheckResponse(
            is_spam=is_spam,
            spam_probability=spam_prob,
            confidence=confidence,
            processing_time_ms=processing_time,
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[SpamCheckResponse])
async def predict_spam_batch(request: BatchSpamCheckRequest):
    """Predict spam for multiple messages"""
    results = []
    
    for msg_request in request.messages:
        try:
            response = await predict_spam(msg_request)
            results.append(response)
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Add error response
            results.append(SpamCheckResponse(
                is_spam=False,
                spam_probability=0.5,
                confidence=0.1,
                processing_time_ms=0.0,
                user_id=msg_request.user_id
            ))
    
    return results

@app.post("/train")
async def submit_training_data(request: TrainingDataRequest):
    """Submit training data (simplified - just log for now)"""
    global training_data_count
    
    training_data_count += 1
    
    logger.info(f"Training data received: user={request.user_id}, is_spam={request.is_spam}, message_length={len(request.message)}")
    
    return {"status": "Training data submitted successfully"}

@app.get("/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    """Get model statistics"""
    uptime = time.time() - start_time
    
    return ModelStatsResponse(
        total_messages_trained=training_data_count,
        spam_messages=spam_detected,
        ham_messages=total_checks - spam_detected,
        uptime_seconds=uptime
    )

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
    uptime = time.time() - start_time
    
    return {
        "total_requests": total_checks,
        "spam_detected": spam_detected,
        "ham_detected": total_checks - spam_detected,
        "spam_rate": spam_detected / total_checks if total_checks > 0 else 0,
        "uptime_seconds": uptime,
        "requests_per_second": total_checks / uptime if uptime > 0 else 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
