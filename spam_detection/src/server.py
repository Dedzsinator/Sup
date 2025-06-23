"""
FastAPI server for hyperoptimized spam detection microservice

This module provides a high-performance REST API for spam detection with:
- Real-time spam classification
- User-specific learning and adaptation
- Model management and monitoring
- Batch processing capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import logging
import asyncio
import time
from datetime import datetime
import json
import os
from contextlib import asynccontextmanager
import uvicorn

from classifier import HyperoptimizedBayesClassifier
from trainer import SpamDetectionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Global classifier instance
classifier = None
trainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global classifier, trainer
    
    logger.info("Starting spam detection microservice...")
    
    # Initialize trainer
    trainer = SpamDetectionTrainer()
    
    # Try to load existing model
    model_path = trainer.load_best_model()
    if model_path:
        classifier = trainer.classifier
        logger.info("Loaded existing model")
    else:
        # Initialize new classifier
        classifier = HyperoptimizedBayesClassifier()
        logger.info("Initialized new classifier")
    
    yield
    
    logger.info("Shutting down spam detection microservice...")

# Initialize FastAPI app
app = FastAPI(
    title="Hyperoptimized Spam Detection API",
    description="Advanced spam detection microservice with user-specific learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SpamCheckRequest(BaseModel):
    """Request model for spam detection"""
    message: str
    user_id: str
    timestamp: Optional[datetime] = None
    
    @validator('message')
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @validator('user_id')
    def user_id_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        return v.strip()

class SpamCheckResponse(BaseModel):
    """Response model for spam detection"""
    is_spam: bool
    spam_probability: float
    confidence: float
    processing_time_ms: float
    user_id: str
    message_id: Optional[str] = None

class BatchSpamCheckRequest(BaseModel):
    """Request model for batch spam detection"""
    messages: List[SpamCheckRequest]
    
    @validator('messages')
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError('Messages list cannot be empty')
        if len(v) > 100:  # Limit batch size
            raise ValueError('Batch size cannot exceed 100 messages')
        return v

class TrainingDataRequest(BaseModel):
    """Request model for training data submission"""
    message: str
    user_id: str
    is_spam: bool
    timestamp: Optional[datetime] = None

class BatchTrainingRequest(BaseModel):
    """Request model for batch training"""
    training_data: List[TrainingDataRequest]
    retrain_model: bool = False

class ModelStatsResponse(BaseModel):
    """Response model for model statistics"""
    total_messages_trained: int
    spam_messages: int
    ham_messages: int
    spam_prior: float
    vocab_size: int
    user_profiles: int
    cache_size: int
    feature_cache_size: int
    uptime_seconds: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    timestamp: datetime
    model_loaded: bool
    
# Global variables for metrics
start_time = time.time()
request_count = 0
error_count = 0

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (implement your authentication logic)"""
    # For demo purposes, accept any token
    # In production, implement proper JWT validation
    if not credentials.token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.token

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global classifier
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        model_loaded=classifier is not None
    )

@app.post("/predict", response_model=SpamCheckResponse)
async def predict_spam(
    request: SpamCheckRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Predict if a message is spam
    """
    global classifier, request_count, error_count
    
    start_time = time.time()
    request_count += 1
    
    try:
        if classifier is None:
            error_count += 1
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Use current time if timestamp not provided
        timestamp = request.timestamp or datetime.now()
        
        # Get prediction
        spam_prob, confidence = classifier.predict(
            request.message,
            request.user_id,
            timestamp
        )
        
        is_spam = spam_prob > 0.5
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            request.user_id,
            request.message,
            is_spam,
            spam_prob,
            confidence,
            processing_time
        )
        
        return SpamCheckResponse(
            is_spam=is_spam,
            spam_probability=spam_prob,
            confidence=confidence,
            processing_time_ms=processing_time,
            user_id=request.user_id
        )
        
    except Exception as e:
        error_count += 1
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[SpamCheckResponse])
async def predict_spam_batch(
    request: BatchSpamCheckRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Predict spam for multiple messages
    """
    global classifier, request_count, error_count
    
    batch_start_time = time.time()
    request_count += len(request.messages)
    
    try:
        if classifier is None:
            error_count += len(request.messages)
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        
        for msg_request in request.messages:
            start_time = time.time()
            
            # Use current time if timestamp not provided
            timestamp = msg_request.timestamp or datetime.now()
            
            # Get prediction
            spam_prob, confidence = classifier.predict(
                msg_request.message,
                msg_request.user_id,
                timestamp
            )
            
            is_spam = spam_prob > 0.5
            processing_time = (time.time() - start_time) * 1000
            
            results.append(SpamCheckResponse(
                is_spam=is_spam,
                spam_probability=spam_prob,
                confidence=confidence,
                processing_time_ms=processing_time,
                user_id=msg_request.user_id
            ))
        
        # Log batch prediction
        batch_time = (time.time() - batch_start_time) * 1000
        background_tasks.add_task(
            log_batch_prediction,
            len(request.messages),
            batch_time
        )
        
        return results
        
    except Exception as e:
        error_count += len(request.messages)
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def submit_training_data(
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Submit training data for model improvement
    """
    global classifier
    
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Use current time if timestamp not provided
        timestamp = request.timestamp or datetime.now()
        
        # Add to training queue (background processing)
        background_tasks.add_task(
            process_training_data,
            request.message,
            request.user_id,
            request.is_spam,
            timestamp
        )
        
        return {"status": "Training data submitted successfully"}
        
    except Exception as e:
        logger.error(f"Training data submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/batch")
async def submit_batch_training(
    request: BatchTrainingRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Submit batch training data
    """
    global classifier, trainer
    
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Process training data
        messages = []
        labels = []
        user_ids = []
        timestamps = []
        
        for data in request.training_data:
            messages.append(data.message)
            labels.append(data.is_spam)
            user_ids.append(data.user_id)
            timestamps.append(data.timestamp or datetime.now())
        
        if request.retrain_model:
            # Full retraining in background
            background_tasks.add_task(
                retrain_model,
                messages,
                labels,
                user_ids,
                timestamps
            )
        else:
            # Incremental training
            background_tasks.add_task(
                incremental_training,
                messages,
                labels,
                user_ids,
                timestamps
            )
        
        return {
            "status": "Batch training submitted successfully",
            "message_count": len(messages),
            "retrain_model": request.retrain_model
        }
        
    except Exception as e:
        logger.error(f"Batch training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=ModelStatsResponse)
async def get_model_stats(token: str = Depends(verify_token)):
    """
    Get model statistics and performance metrics
    """
    global classifier, start_time
    
    try:
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        stats = classifier.get_model_stats()
        uptime = time.time() - start_time
        
        return ModelStatsResponse(
            total_messages_trained=stats['total_messages_trained'],
            spam_messages=stats['spam_messages'],
            ham_messages=stats['ham_messages'],
            spam_prior=stats['spam_prior'],
            vocab_size=stats['vocab_size'],
            user_profiles=stats['user_profiles'],
            cache_size=stats['cache_size'],
            feature_cache_size=stats['feature_cache_size'],
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics(token: str = Depends(verify_token)):
    """
    Get API performance metrics
    """
    global request_count, error_count, start_time
    
    uptime = time.time() - start_time
    error_rate = error_count / request_count if request_count > 0 else 0
    
    return {
        "total_requests": request_count,
        "total_errors": error_count,
        "error_rate": error_rate,
        "uptime_seconds": uptime,
        "requests_per_second": request_count / uptime if uptime > 0 else 0
    }

@app.post("/retrain")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """
    Trigger model retraining with all available data
    """
    global trainer
    
    try:
        if trainer is None:
            raise HTTPException(status_code=503, detail="Trainer not available")
        
        # Start retraining in background
        background_tasks.add_task(full_model_retrain)
        
        return {"status": "Model retraining started"}
        
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions

async def log_prediction(user_id: str, message: str, is_spam: bool, 
                        spam_prob: float, confidence: float, processing_time: float):
    """Log prediction for monitoring and analysis"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "message_length": len(message),
        "is_spam": is_spam,
        "spam_probability": spam_prob,
        "confidence": confidence,
        "processing_time_ms": processing_time
    }
    
    # Log to file or monitoring system
    logger.info(f"Prediction: {json.dumps(log_entry)}")

async def log_batch_prediction(batch_size: int, batch_time: float):
    """Log batch prediction metrics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "batch_processing_time_ms": batch_time,
        "avg_time_per_message": batch_time / batch_size
    }
    
    logger.info(f"Batch prediction: {json.dumps(log_entry)}")

async def process_training_data(message: str, user_id: str, is_spam: bool, timestamp: datetime):
    """Process individual training data point"""
    global classifier
    
    try:
        # Online learning - update model incrementally
        classifier.train([message], [is_spam], [user_id], [timestamp])
        logger.info(f"Processed training data for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error processing training data: {e}")

async def incremental_training(messages: List[str], labels: List[bool],
                             user_ids: List[str], timestamps: List[datetime]):
    """Perform incremental training with batch data"""
    global classifier
    
    try:
        classifier.train(messages, labels, user_ids, timestamps)
        logger.info(f"Incremental training completed with {len(messages)} messages")
        
    except Exception as e:
        logger.error(f"Error in incremental training: {e}")

async def retrain_model(messages: List[str], labels: List[bool],
                       user_ids: List[str], timestamps: List[datetime]):
    """Retrain model with new data"""
    global classifier, trainer
    
    try:
        # Load existing training data and combine with new data
        existing_messages, existing_labels, existing_users, existing_timestamps = trainer.load_training_data()
        
        all_messages = existing_messages + messages
        all_labels = existing_labels + labels
        all_users = existing_users + user_ids
        all_timestamps = existing_timestamps + timestamps
        
        # Train new model
        results = trainer.train_model(all_messages, all_labels, all_users, all_timestamps)
        
        # Update global classifier
        classifier = trainer.classifier
        
        logger.info(f"Model retraining completed. F1 score: {results['test_metrics']['f1']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")

async def full_model_retrain():
    """Perform full model retraining"""
    global classifier, trainer
    
    try:
        # Load all available training data
        messages, labels, user_ids, timestamps = trainer.load_training_data()
        
        if len(messages) > 0:
            # Train new model
            results = trainer.train_model(messages, labels, user_ids, timestamps)
            
            # Update global classifier
            classifier = trainer.classifier
            
            logger.info(f"Full retraining completed. F1 score: {results['test_metrics']['f1']:.3f}")
        else:
            logger.warning("No training data available for retraining")
        
    except Exception as e:
        logger.error(f"Error in full retraining: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
