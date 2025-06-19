#!/usr/bin/env python3
"""
FastAPI server for the intelligent autocomplete system.

This server provides REST API endpoints that interface with the autocomplete
system components (Trie, semantic search, AI ranking, and text generation).
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the src directory to the path to import our modules
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

try:
    from pipeline import AutocompletePipeline, PipelineConfig
except ImportError as e:
    logging.error(f"Failed to import autocomplete modules: {e}")
    logging.error("Make sure all dependencies are installed and the src directory is properly set up")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request/Response models
class AutocompleteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=10)

class CompletionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    max_length: int = Field(default=50, ge=1, le=200)

class TrainingMessage(BaseModel):
    text: str
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    timestamp: Optional[str] = None

class TrainingRequest(BaseModel):
    messages: List[TrainingMessage]

class AutocompleteResponse(BaseModel):
    suggestions: List[str]
    latency_ms: float
    source: str

class CompletionResponse(BaseModel):
    completion: str
    confidence: float
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]

class StatsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    cache_hit_rate: float
    trie_size: int
    index_size: int
    memory_usage_mb: float

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Autocomplete API",
    description="REST API for the hybrid autocomplete system with Trie + AI components",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline: Optional[AutocompletePipeline] = None
start_time = time.time()
request_count = 0
total_latency = 0.0

# Initialize the autocomplete pipeline
async def initialize_pipeline():
    """Initialize the autocomplete pipeline with sample data."""
    global pipeline
    
    try:
        logger.info("Initializing autocomplete pipeline...")
        
        # Create pipeline with minimal configuration
        config = PipelineConfig(
            use_trie=True,
            use_semantic_search=False,  # Disable for now to avoid model loading issues
            use_ranking=False,
            use_generation=False,
            max_suggestions=10
        )
        
        pipeline = AutocompletePipeline(config)
        
        # Check if trained models exist and load them
        models_dir = current_dir / "models"
        if models_dir.exists():
            logger.info(f"Found models directory: {models_dir}")
            await pipeline.initialize(models_dir)
        else:
            logger.info("No models directory found, using default initialization")
            await pipeline.initialize()
        
        # Load sample chat data for initial training
        sample_phrases = [
            "Hey, how are you?",
            "Good morning everyone!",
            "What's up?",
            "Thanks for the help",
            "See you later",
            "Have a great day!",
            "Nice work on the project",
            "Let's meet tomorrow",
            "I'll be there in 5 minutes",
            "Can you send me the file?",
            "The meeting starts at 3pm",
            "Happy birthday!",
            "Congratulations on your promotion",
            "How was your weekend?",
            "I'm running late",
            "Sorry, I missed your message",
            "Let me know if you need anything",
            "Great job team!",
            "I agree with your suggestion",
            "That sounds like a good plan",
            "Could you clarify that?",
            "I'll get back to you soon",
            "Thanks for the quick response",
            "Have you seen the latest update?",
            "I'm excited about this project",
            "When is the deadline?",
            "Let's schedule a call",
            "I'll send you the details",
            "Perfect, that works for me",
            "I'm looking forward to it",
        ]
        
        # Build the trie with sample data
        if pipeline.trie:
            for phrase in sample_phrases:
                pipeline.trie.insert(phrase)
        
        logger.info(f"Pipeline initialized successfully with {len(sample_phrases)} phrases")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline when the server starts."""
    await initialize_pipeline()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global pipeline, start_time
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    uptime = time.time() - start_time
    
    # Check component health
    components = {
        "trie": "healthy" if pipeline.trie else "unhealthy",
        "embedder": "healthy" if pipeline.embedder else "unhealthy",
        "indexer": "healthy" if pipeline.indexer else "unhealthy",
        "ranker": "healthy" if pipeline.ranker else "unhealthy",
        "generator": "healthy" if pipeline.generator else "unhealthy",
    }
    
    status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime_seconds=uptime,
        components=components
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    global pipeline, request_count, total_latency
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Calculate statistics
    avg_latency = total_latency / max(request_count, 1)
    cache_stats = pipeline.get_cache_stats()
    
    import psutil
    import sys
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    return StatsResponse(
        total_requests=request_count,
        avg_latency_ms=avg_latency * 1000,
        cache_hit_rate=cache_stats.get("hit_rate", 0.0),
        trie_size=len(pipeline.trie) if pipeline.trie else 0,
        index_size=pipeline.indexer.get_size() if pipeline.indexer else 0,
        memory_usage_mb=memory_mb
    )

@app.post("/suggest", response_model=AutocompleteResponse)
async def get_suggestions(request: AutocompleteRequest):
    """Get autocomplete suggestions for the given text."""
    global pipeline, request_count, total_latency
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start = time.time()
    request_count += 1
    
    try:
        # Get suggestions from the pipeline
        suggestions = await pipeline.get_suggestions(
            text=request.text,
            user_id=request.user_id,
            room_id=request.room_id,
            limit=request.limit
        )
        
        latency = time.time() - start
        total_latency += latency
        
        # Determine primary source
        source = "hybrid"  # Our system uses multiple sources
        
        return AutocompleteResponse(
            suggestions=suggestions,
            latency_ms=latency * 1000,
            source=source
        )
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete", response_model=CompletionResponse)
async def get_completion(request: CompletionRequest):
    """Get text completion for the given input."""
    global pipeline, request_count, total_latency
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start = time.time()
    request_count += 1
    
    try:
        # Get completion from the pipeline
        completion, confidence = await pipeline.get_completion(
            text=request.text,
            user_id=request.user_id,
            room_id=request.room_id,
            max_length=request.max_length
        )
        
        latency = time.time() - start
        total_latency += latency
        
        return CompletionResponse(
            completion=completion,
            confidence=confidence,
            latency_ms=latency * 1000
        )
        
    except Exception as e:
        logger.error(f"Error getting completion: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train the models with new chat messages."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Extract phrases from messages
        phrases = [msg.text for msg in request.messages]
        
        # Add training task to background
        background_tasks.add_task(train_with_phrases, phrases)
        
        return {"status": "success", "message": f"Training started with {len(phrases)} messages"}
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

async def train_with_phrases(phrases: List[str]):
    """Background task to train the pipeline with new phrases."""
    global pipeline
    
    try:
        logger.info(f"Starting background training with {len(phrases)} phrases")
        
        # Add phrases to trie
        for phrase in phrases:
            if phrase and len(phrase.strip()) > 0:
                pipeline.trie.insert(phrase.strip())
        
        # Update semantic indices (if implemented)
        if hasattr(pipeline, 'update_indices'):
            await pipeline.update_indices(phrases)
        
        logger.info("Background training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}")
        logger.error(traceback.format_exc())

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Intelligent Autocomplete API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "suggestions": "POST /suggest",
            "completion": "POST /complete",
            "training": "POST /train",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting autocomplete API server on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
