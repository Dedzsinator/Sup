"""
Configuration management for spam detection microservice
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    vocab_size: int = 50000
    user_vector_dim: int = 512
    temporal_decay: float = 0.95
    confidence_threshold: float = 0.8
    batch_size: int = 100
    learning_rate: float = 0.01

@dataclass
class ServerConfig:
    """Server configuration parameters"""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "info"
    cors_origins: str = "*"
    api_key: str = "your-secret-api-key"

@dataclass
class DataConfig:
    """Data configuration parameters"""
    data_dir: str = "/app/data"
    models_dir: str = "/app/models"
    config_dir: str = "/app/config"
    max_message_length: int = 10000
    min_message_length: int = 3

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    
    model_config = ModelConfig(
        vocab_size=int(os.getenv("VOCAB_SIZE", 50000)),
        user_vector_dim=int(os.getenv("USER_VECTOR_DIM", 512)),
        temporal_decay=float(os.getenv("TEMPORAL_DECAY", 0.95)),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", 0.8)),
        batch_size=int(os.getenv("BATCH_SIZE", 100)),
        learning_rate=float(os.getenv("LEARNING_RATE", 0.01))
    )
    
    server_config = ServerConfig(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        workers=int(os.getenv("WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info"),
        cors_origins=os.getenv("CORS_ORIGINS", "*"),
        api_key=os.getenv("API_KEY", "your-secret-api-key")
    )
    
    data_config = DataConfig(
        data_dir=os.getenv("DATA_DIR", "/app/data"),
        models_dir=os.getenv("MODELS_DIR", "/app/models"),
        config_dir=os.getenv("CONFIG_DIR", "/app/config"),
        max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", 10000)),
        min_message_length=int(os.getenv("MIN_MESSAGE_LENGTH", 3))
    )
    
    return {
        "model": model_config,
        "server": server_config,
        "data": data_config
    }
