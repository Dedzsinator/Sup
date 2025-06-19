"""
Intelligent Autocomplete System for Real-Time Chat Applications

This package provides a hybrid autocomplete engine that combines:
- Trie-based fast prefix matching
- Semantic vector search using sentence embeddings  
- AI-based ranking with PyTorch neural models
- Optional transformer-based completion

Modules:
    trie: Fast prefix tree implementation
    embedder: Sentence embedding and contrastive learning
    indexer: FAISS-based approximate nearest neighbor search
    ranker: PyTorch reranking model
    generator: Optional transformer-based completion
    pipeline: Full inference pipeline
"""

__version__ = "0.1.0"
__author__ = "Sup Chat Application Team"

from .trie import Trie, TrieNode
from .embedder import SentenceEmbedder, EmbedderConfig, ContrastiveDataset, ContrastiveLoss
from .indexer import VectorIndexer, IndexConfig, HybridIndexer
from .ranker import RankingModel, RankerConfig, RankingDataset, extract_ranking_features
from .generator import ChatGenerator, GeneratorConfig, ChatDataset
from .pipeline import (
    AutocompletePipeline, 
    PipelineConfig, 
    AutocompleteRequest, 
    AutocompleteResponse,
    AutocompleteSuggestion
)

__all__ = [
    # Core components
    "Trie",
    "TrieNode",
    "SentenceEmbedder", 
    "EmbedderConfig",
    "VectorIndexer",
    "IndexConfig", 
    "HybridIndexer",
    "RankingModel",
    "RankerConfig",
    "ChatGenerator",
    "GeneratorConfig",
    
    # Pipeline
    "AutocompletePipeline",
    "PipelineConfig",
    "AutocompleteRequest",
    "AutocompleteResponse", 
    "AutocompleteSuggestion",
    
    # Training utilities
    "ContrastiveDataset",
    "ContrastiveLoss",
    "RankingDataset",
    "ChatDataset",
    "extract_ranking_features",
]
