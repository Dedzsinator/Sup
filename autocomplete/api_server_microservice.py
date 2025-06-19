#!/usr/bin/env python3
"""
Microservice API server for the intelligent autocomplete system.

This microservice provides REST API endpoints that interface with the autocomplete
system components specifically designed for the Elixir backend integration.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
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

# Request/Response models for Elixir backend compatibility
class AutocompleteRequest(BaseModel):
    text: str = Field(..., description="The input text to get suggestions for")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    room_id: Optional[str] = Field(None, description="Room ID for context")
    limit: Optional[int] = Field(5, description="Maximum number of suggestions")
    max_suggestions: Optional[int] = Field(None, description="Alternative parameter name for limit")

class CompletionRequest(BaseModel):
    text: str = Field(..., description="The input text to complete")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    room_id: Optional[str] = Field(None, description="Room ID for context")
    max_length: Optional[int] = Field(50, description="Maximum length of completion")

class TrainingMessage(BaseModel):
    text: str
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    timestamp: Optional[str] = None

class TrainingRequest(BaseModel):
    messages: List[TrainingMessage] = Field(..., description="Messages to train on")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    pipeline_loaded: bool

class StatsResponse(BaseModel):
    requests_processed: int
    average_latency_ms: float
    trie_size: int
    model_status: Dict[str, str]

# Initialize FastAPI app
app = FastAPI(
    title="Autocomplete Microservice",
    description="Microservice for intelligent autocomplete system integrated with Elixir backend",
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

@app.on_event("startup")
async def startup_event():
    """Initialize the autocomplete pipeline on startup."""
    global pipeline
    try:
        logger.info("Initializing autocomplete pipeline...")
        
        # Create pipeline configuration
        config = PipelineConfig(
            data_dir=current_dir / "data",
            models_dir=current_dir / "models",
            trie_max_depth=10,
            embedding_dim=128,
            top_k_trie=10,
            top_k_semantic=5,
            generation_max_length=100
        )
        
        # Initialize pipeline
        pipeline = AutocompletePipeline(config)
        await asyncio.to_thread(pipeline.initialize)
        
        logger.info("Autocomplete pipeline initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        # Continue running for health checks, but mark as unhealthy
        pipeline = None

@app.get("/health")
async def health_check():
    """Health check endpoint for the microservice."""
    global start_time, pipeline
    
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
        pipeline_loaded=pipeline is not None
    )

@app.get("/stats")
async def get_stats():
    """Get statistics about the autocomplete service."""
    global request_count, total_latency, pipeline
    
    avg_latency = (total_latency / request_count) if request_count > 0 else 0
    
    model_status = {}
    trie_size = 0
    
    if pipeline:
        try:
            trie_size = len(pipeline.trie.root.children) if hasattr(pipeline.trie, 'root') else 0
            model_status = {
                "trie": "loaded",
                "embedder": "loaded" if hasattr(pipeline, 'embedder') else "not_loaded",
                "ranker": "loaded" if hasattr(pipeline, 'ranker') else "not_loaded",
                "generator": "loaded" if hasattr(pipeline, 'generator') else "not_loaded"
            }
        except Exception as e:
            logger.warning(f"Error getting model status: {e}")
            model_status = {"error": str(e)}
    
    return StatsResponse(
        requests_processed=request_count,
        average_latency_ms=avg_latency,
        trie_size=trie_size,
        model_status=model_status
    )

@app.post("/suggest")
async def get_suggestions(request: AutocompleteRequest):
    """Get autocomplete suggestions for the given text."""
    global pipeline, request_count, total_latency
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time_req = time.time()
    
    try:
        # Handle both 'limit' and 'max_suggestions' parameters
        limit = request.limit
        if request.max_suggestions is not None:
            limit = request.max_suggestions
        
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            return []
        
        if len(request.text) > 500:
            raise HTTPException(status_code=400, detail="Text too long (max 500 characters)")
        
        if limit > 20:
            raise HTTPException(status_code=400, detail="Limit too high (max 20)")
        
        # Get suggestions from pipeline
        suggestions = await asyncio.to_thread(
            pipeline.get_suggestions,
            request.text.strip(),
            max_suggestions=limit,
            user_context={
                "user_id": request.user_id,
                "room_id": request.room_id
            }
        )
        
        # Update metrics
        latency = (time.time() - start_time_req) * 1000
        request_count += 1
        total_latency += latency
        
        # Return simple list for Elixir backend compatibility
        return suggestions[:limit] if suggestions else []
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/complete")
async def get_completion(request: CompletionRequest):
    """Get text completion for the given input."""
    global pipeline, request_count, total_latency
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    start_time_req = time.time()
    
    try:
        # Validate input
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text is required")
        
        if len(request.text) > 500:
            raise HTTPException(status_code=400, detail="Text too long (max 500 characters)")
        
        # Get completion from pipeline
        completion = await asyncio.to_thread(
            pipeline.generate_completion,
            request.text.strip(),
            max_length=request.max_length,
            user_context={
                "user_id": request.user_id,
                "room_id": request.room_id
            }
        )
        
        # Update metrics
        latency = (time.time() - start_time_req) * 1000
        request_count += 1
        total_latency += latency
        
        # Return simple completion string for Elixir backend
        return {"completion": completion or ""}
        
    except Exception as e:
        logger.error(f"Error getting completion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train the autocomplete system with new messages."""
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Convert messages to training format
        training_data = []
        for msg in request.messages:
            training_data.append({
                "text": msg.text,
                "metadata": {
                    "user_id": msg.user_id,
                    "room_id": msg.room_id,
                    "timestamp": msg.timestamp
                }
            })
        
        # Start training in background
        asyncio.create_task(
            asyncio.to_thread(pipeline.train_incremental, training_data)
        )
        
        return {"status": "success", "message": "Training started", "messages_count": len(training_data)}
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with microservice information."""
    return {
        "name": "Autocomplete Microservice",
        "version": "1.0.0",
        "status": "running",
        "pipeline_loaded": pipeline is not None,
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
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting autocomplete microservice on {host}:{port}")
    logger.info(f"Workers: {workers}")
    
    # Run the server
    uvicorn.run(
        "api_server_microservice:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )