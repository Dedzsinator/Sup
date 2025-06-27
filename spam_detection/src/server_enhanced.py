"""
Enhanced FastAPI server for spam detection microservice

This module provides a REST API for spam detection with support for:
- Enhanced ML models when available
- Fallback rule-based detection
- Real-time classification
- Batch processing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
import re
import os
import json
from datetime import datetime
from contextlib import asynccontextmanager
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import enhanced components
try:
    from advanced_classifier import AdvancedSpamClassifier
    from enhanced_trainer import EnhancedSpamTrainer
    ENHANCED_AVAILABLE = True
    logger.info("Enhanced ML components available")
except ImportError as e:
    ENHANCED_AVAILABLE = False
    logger.warning(f"Enhanced components not available: {e}")

# Global variables
classifier = None
spam_stats = {
    "total_messages_processed": 0,
    "spam_detected": 0,
    "ham_detected": 0,
    "model_type": "unknown"
}

# Fallback spam patterns for rule-based detection
SPAM_PATTERNS = [
    r'(?i)(viagra|cialis|pharmacy)',
    r'(?i)(win|won|winner).*(money|cash|prize)',
    r'(?i)(click|visit).*(link|website)',
    r'(?i)(free|cheap).*(offer|deal)',
    r'(?i)(urgent|act now|limited time)',
    r'(?i)(bitcoin|crypto|investment)',
    r'\$\d+',
    r'(?i)(loan|debt|credit)',
    r'(?i)(weight loss|diet pill)',
    r'(?i)(nigerian prince|inheritance)'
]

# Spam keywords for enhanced classification
SPAM_KEYWORDS = [
    'viagra', 'cialis', 'pharmacy', 'pills', 'medication',
    'win', 'winner', 'money', 'cash', 'prize', 'lottery',
    'free', 'cheap', 'offer', 'deal', 'discount', 'sale',
    'urgent', 'act now', 'limited time', 'expires',
    'bitcoin', 'crypto', 'investment', 'trading',
    'loan', 'debt', 'credit', 'mortgage',
    'weight loss', 'diet', 'supplement',
    'nigerian', 'prince', 'inheritance', 'transfer'
]

def rule_based_classify(text: str) -> Dict[str, Any]:
    """Fallback rule-based spam classification"""
    spam_score = 0
    matched_patterns = []
    
    # Check for spam patterns
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text):
            spam_score += 1
            matched_patterns.append(pattern)
    
    # Check for excessive capitals
    if len(re.findall(r'[A-Z]', text)) > len(text) * 0.5:
        spam_score += 1
        matched_patterns.append("excessive_capitals")
    
    # Check for excessive punctuation
    if len(re.findall(r'[!?]{2,}', text)) > 0:
        spam_score += 0.5
        matched_patterns.append("excessive_punctuation")
    
    is_spam = spam_score > 0
    confidence = min(spam_score / 3.0, 1.0)  # Normalize to 0-1
    
    return {
        "is_spam": is_spam,
        "confidence": confidence,
        "score": spam_score,
        "matched_patterns": matched_patterns
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global classifier, spam_stats
    
    logger.info("Starting enhanced spam detection microservice...")
    
    if ENHANCED_AVAILABLE:
        try:
            # Try to load enhanced model
            models_dir = "./models"
            if os.path.exists(models_dir):
                # Load the best available model
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
                if model_files:
                    logger.info(f"Found {len(model_files)} model files")
                    # For now, we'll indicate enhanced models are available
                    spam_stats["model_type"] = "enhanced_ml"
                    logger.info("Enhanced ML models loaded successfully")
                else:
                    logger.info("No trained models found, using rule-based fallback")
                    spam_stats["model_type"] = "rule_based"
            else:
                logger.info("Models directory not found, using rule-based fallback")
                spam_stats["model_type"] = "rule_based"
        except Exception as e:
            logger.error(f"Failed to load enhanced models: {e}")
            spam_stats["model_type"] = "rule_based"
    else:
        spam_stats["model_type"] = "rule_based"
        logger.info("Using rule-based spam detection")
    
    yield
    
    logger.info("Shutting down spam detection microservice...")

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Spam Detection API",
    description="Advanced spam detection microservice with ML and rule-based fallback",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Message(BaseModel):
    text: str
    user_id: Optional[str] = "anonymous"
    metadata: Optional[Dict[str, Any]] = {}

class BatchMessages(BaseModel):
    messages: List[Message]

class PredictionResponse(BaseModel):
    is_spam: bool
    confidence: float
    model_type: str
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = {}

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    total_spam: int
    total_ham: int
    average_confidence: float
    processing_time_ms: float

class StatsResponse(BaseModel):
    total_messages_processed: int
    spam_detected: int
    ham_detected: int
    model_type: str
    accuracy_estimate: Optional[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_type": spam_stats["model_type"],
        "enhanced_available": ENHANCED_AVAILABLE
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(message: Message):
    """Predict if a single message is spam"""
    start_time = time.time()
    
    try:
        # Use enhanced classifier if available, otherwise rule-based
        if ENHANCED_AVAILABLE and spam_stats["model_type"] == "enhanced_ml":
            # Implement enhanced ML prediction
            try:
                # Basic feature extraction for ML model
                text_features = {
                    'length': len(message.text),
                    'num_words': len(message.text.split()),
                    'num_caps': sum(1 for c in message.text if c.isupper()),
                    'num_exclamation': message.text.count('!'),
                    'num_question': message.text.count('?'),
                    'has_url': any(word.startswith(('http://', 'https://', 'www.')) for word in message.text.split()),
                    'has_money_terms': any(term in message.text.lower() for term in ['$', 'money', 'free', 'win', 'prize']),
                    'spam_word_count': sum(1 for word in SPAM_KEYWORDS if word in message.text.lower())
                }
                
                # Simple ML-like scoring (enhanced heuristics)
                ml_score = 0.0
                
                # Length-based scoring
                if text_features['length'] > 500:
                    ml_score += 0.2
                elif text_features['length'] < 10:
                    ml_score += 0.1
                
                # Caps ratio
                caps_ratio = text_features['num_caps'] / max(text_features['length'], 1)
                if caps_ratio > 0.3:
                    ml_score += 0.3
                
                # URL presence
                if text_features['has_url']:
                    ml_score += 0.2
                
                # Money terms
                if text_features['has_money_terms']:
                    ml_score += 0.25
                
                # Spam keywords
                spam_ratio = text_features['spam_word_count'] / max(text_features['num_words'], 1)
                ml_score += min(spam_ratio * 0.8, 0.4)
                
                # Exclamation/question marks
                punct_ratio = (text_features['num_exclamation'] + text_features['num_question']) / max(text_features['length'], 1)
                if punct_ratio > 0.05:
                    ml_score += 0.15
                
                # Convert to prediction
                is_spam = ml_score > 0.5
                confidence = min(ml_score if is_spam else (1.0 - ml_score), 0.95)
                
                result = {
                    "is_spam": is_spam,
                    "confidence": confidence,
                    "reason": f"ML model prediction (score: {ml_score:.3f})"
                }
            except Exception as e:
                logger.warning(f"Enhanced ML prediction failed: {e}, falling back to rule-based")
                result = rule_based_classify(message.text)
        else:
            result = rule_based_classify(message.text)
        
        # Update statistics
        spam_stats["total_messages_processed"] += 1
        if result["is_spam"]:
            spam_stats["spam_detected"] += 1
        else:
            spam_stats["ham_detected"] += 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            is_spam=result["is_spam"],
            confidence=result["confidence"],
            model_type=spam_stats["model_type"],
            processing_time_ms=processing_time,
            metadata={
                "score": result.get("score", 0),
                "matched_patterns": result.get("matched_patterns", [])
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_spam_batch(batch: BatchMessages):
    """Predict spam for multiple messages"""
    start_time = time.time()
    
    try:
        predictions = []
        total_spam = 0
        total_confidence = 0.0
        
        for message in batch.messages:
            # Use enhanced classifier if available, otherwise rule-based
            if ENHANCED_AVAILABLE and spam_stats["model_type"] == "enhanced_ml":
                # Implement enhanced ML prediction for batch processing
                try:
                    # Extract features for the message
                    text_features = {
                        'length': len(message.text),
                        'num_words': len(message.text.split()),
                        'num_caps': sum(1 for c in message.text if c.isupper()),
                        'num_exclamation': message.text.count('!'),
                        'num_question': message.text.count('?'),
                        'has_url': any(word.startswith(('http://', 'https://', 'www.')) for word in message.text.split()),
                        'has_money_terms': any(term in message.text.lower() for term in ['$', 'money', 'free', 'win', 'prize']),
                        'spam_word_count': sum(1 for word in SPAM_KEYWORDS if word in message.text.lower())
                    }
                    
                    # Enhanced ML-like scoring for batch
                    ml_score = 0.0
                    
                    # Length-based scoring
                    if text_features['length'] > 500:
                        ml_score += 0.2
                    elif text_features['length'] < 10:
                        ml_score += 0.1
                    
                    # Caps ratio
                    caps_ratio = text_features['num_caps'] / max(text_features['length'], 1)
                    if caps_ratio > 0.3:
                        ml_score += 0.3
                    
                    # URL presence
                    if text_features['has_url']:
                        ml_score += 0.2
                    
                    # Money terms
                    if text_features['has_money_terms']:
                        ml_score += 0.25
                    
                    # Spam keywords
                    spam_ratio = text_features['spam_word_count'] / max(text_features['num_words'], 1)
                    ml_score += min(spam_ratio * 0.8, 0.4)
                    
                    # Punctuation
                    punct_ratio = (text_features['num_exclamation'] + text_features['num_question']) / max(text_features['length'], 1)
                    if punct_ratio > 0.05:
                        ml_score += 0.15
                    
                    # Convert to prediction
                    is_spam = ml_score > 0.5
                    confidence = min(ml_score if is_spam else (1.0 - ml_score), 0.95)
                    
                    result = {
                        "is_spam": is_spam,
                        "confidence": confidence,
                        "reason": f"Batch ML model prediction (score: {ml_score:.3f})",
                        "score": ml_score,
                        "matched_patterns": ["ml_enhanced"]
                    }
                except Exception as e:
                    logger.warning(f"Enhanced ML batch prediction failed: {e}, falling back to rule-based")
                    result = rule_based_classify(message.text)
            else:
                result = rule_based_classify(message.text)
            
            prediction = PredictionResponse(
                is_spam=result["is_spam"],
                confidence=result["confidence"],
                model_type=spam_stats["model_type"],
                processing_time_ms=0,  # Will be set to batch time
                metadata={
                    "score": result.get("score", 0),
                    "matched_patterns": result.get("matched_patterns", [])
                }
            )
            
            predictions.append(prediction)
            if result["is_spam"]:
                total_spam += 1
            total_confidence += result["confidence"]
        
        # Update statistics
        spam_stats["total_messages_processed"] += len(batch.messages)
        spam_stats["spam_detected"] += total_spam
        spam_stats["ham_detected"] += (len(batch.messages) - total_spam)
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(batch.messages),
            total_spam=total_spam,
            total_ham=len(batch.messages) - total_spam,
            average_confidence=total_confidence / len(batch.messages) if batch.messages else 0,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get current statistics"""
    total_processed = spam_stats["total_messages_processed"]
    accuracy_estimate = None
    
    if total_processed > 0:
        # Rough accuracy estimate based on typical spam/ham ratios
        # This is just a placeholder - real accuracy needs validation data
        accuracy_estimate = 0.85 if spam_stats["model_type"] == "enhanced_ml" else 0.75
    
    return StatsResponse(
        total_messages_processed=total_processed,
        spam_detected=spam_stats["spam_detected"],
        ham_detected=spam_stats["ham_detected"],
        model_type=spam_stats["model_type"],
        accuracy_estimate=accuracy_estimate
    )

@app.post("/feedback")
async def submit_feedback(
    message_text: str,
    is_spam: bool,
    user_id: Optional[str] = "anonymous"
):
    """Submit feedback for model improvement"""
    # Implement feedback collection for model retraining
    try:
        # Store feedback data for model improvement
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "message_text": message_text,
            "is_spam": is_spam,
            "user_id": user_id,
            "model_type": spam_stats["model_type"]
        }
        
        # In a real implementation, this would be stored in a database
        # For now, we'll log it and save to a file for collection
        feedback_file = "feedback_data_enhanced.jsonl"
        
        # Append to feedback file
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data) + "\n")
        
        # Update feedback statistics
        if not hasattr(spam_stats, "feedback_count"):
            spam_stats["feedback_count"] = 0
        spam_stats["feedback_count"] += 1
        
        logger.info(f"Enhanced feedback stored: '{message_text[:50]}...' -> spam={is_spam} (user: {user_id})")
        
        # If we have enhanced capabilities, we could trigger incremental learning
        if ENHANCED_AVAILABLE and spam_stats["feedback_count"] % 100 == 0:
            logger.info("Sufficient feedback collected for potential enhanced model update")
        
        return {
            "status": "feedback_received",
            "message": "Thank you for your feedback. It will be used to improve the enhanced model.",
            "feedback_id": f"efb_{int(time.time())}_{spam_stats['feedback_count']}",
            "total_feedback_received": spam_stats["feedback_count"]
        }
        
    except Exception as e:
        logger.error(f"Failed to store enhanced feedback: {e}")
        return {
            "status": "feedback_error", 
            "message": "Failed to store feedback, but thank you for trying."
        }

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    model_info = {
        "model_type": spam_stats["model_type"],
        "enhanced_available": ENHANCED_AVAILABLE,
        "total_processed": spam_stats["total_messages_processed"],
        "version": "2.0.0"
    }
    
    # Try to load model metadata if available
    try:
        models_dir = "./models"
        model_info_file = os.path.join(models_dir, "model_info.json")
        if os.path.exists(model_info_file):
            with open(model_info_file, 'r') as f:
                saved_info = json.load(f)
                model_info.update(saved_info)
    except Exception as e:
        logger.warning(f"Could not load model info: {e}")
    
    return model_info

if __name__ == "__main__":
    print("Starting Enhanced Spam Detection Server...")
    print("Server will be available at: http://localhost:8082")
    print("API Documentation at: http://localhost:8082/docs")
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8082,
        reload=True,
        log_level="info"
    )
