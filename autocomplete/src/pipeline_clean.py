"""
Unified Autocomplete Pipeline

This module combines all components (Trie, Embedder, Indexer, Ranker, Generator)
into a cohesive autocomplete system for real-time chat applications.

Features:
- Multi-modal suggestion generation (prefix, semantic, predictive)
- Intelligent ranking and fusion of suggestions
- Real-time performance optimization
- Personalization and context awareness
- A/B testing and analytics integration
- Configurable components and fallbacks

Author: Generated for Sup Chat Application
"""

import asyncio
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import json

# Import our components
from trie import Trie
from embedder import SentenceEmbedder, EmbedderConfig
from indexer import VectorIndexer, IndexConfig
from ranker import RankingModel, RankerConfig, extract_ranking_features
from generator import ChatGenerator, GeneratorConfig


@dataclass
class PipelineConfig:
    """Configuration for the autocomplete pipeline."""
    # Component configurations
    trie_config: Dict[str, Any] = field(default_factory=lambda: {
        "case_sensitive": False,
        "max_suggestions": 20
    })
    embedder_config: Optional[EmbedderConfig] = None
    indexer_config: Optional[IndexConfig] = None
    ranker_config: Optional[RankerConfig] = None
    generator_config: Optional[GeneratorConfig] = None
    
    # Pipeline behavior
    max_suggestions: int = 10
    min_query_length: int = 1
    use_trie: bool = True
    use_semantic_search: bool = True
    use_ranking: bool = True
    use_generation: bool = False
    
    # Performance settings
    max_concurrent_requests: int = 100
    cache_size: int = 10000
    cache_ttl: int = 3600  # seconds
    timeout: float = 0.1  # seconds
    
    # Weights for suggestion fusion
    trie_weight: float = 0.4
    semantic_weight: float = 0.4
    generation_weight: float = 0.2
    
    # Personalization
    enable_personalization: bool = True
    user_history_size: int = 1000
    
    # Analytics
    enable_analytics: bool = True
    log_suggestions: bool = False


@dataclass
class AutocompleteRequest:
    """Request for autocomplete suggestions."""
    query: str
    user_id: Optional[str] = None
    context: Optional[str] = None
    conversation_id: Optional[str] = None
    max_suggestions: Optional[int] = None
    include_metadata: bool = False
    request_id: Optional[str] = None


@dataclass
class AutocompleteSuggestion:
    """A single autocomplete suggestion."""
    text: str
    score: float
    source: str  # "trie", "semantic", "generation"
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutocompleteResponse:
    """Response with autocomplete suggestions."""
    suggestions: List[AutocompleteSuggestion]
    query: str
    processing_time: float
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SuggestionCache:
    """LRU cache for autocomplete suggestions."""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[AutocompleteResponse, float]] = {}
        self.access_order: deque = deque()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[AutocompleteResponse]:
        """Get cached response if available and not expired."""
        with self._lock:
            if key in self.cache:
                response, timestamp = self.cache[key]
                
                # Check if expired
                if time.time() - timestamp > self.ttl:
                    del self.cache[key]
                    self.access_order.remove(key)
                    return None
                
                # Update access order
                self.access_order.remove(key)
                self.access_order.append(key)
                
                return response
            
            return None
    
    def put(self, key: str, response: AutocompleteResponse):
        """Cache response with timestamp."""
        with self._lock:
            # Remove oldest if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = (response, time.time())
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()


class AutocompletePipeline:
    """
    Main autocomplete pipeline that orchestrates all components.
    
    This class provides the primary interface for generating autocomplete
    suggestions by combining prefix matching, semantic search, and generation.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the autocomplete pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._init_components()
        
        # Performance tracking
        self.stats = {
            'requests_total': 0,
            'requests_cached': 0,
            'avg_processing_time': 0.0,
            'error_count': 0,
            'component_times': defaultdict(list)
        }
        
        # Caching
        self.cache = SuggestionCache(config.cache_size, config.cache_ttl)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        
        # User personalization data
        self.user_histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.user_history_size)
        )
        
        self.logger.info("Autocomplete pipeline initialized")
    
    def _init_components(self):
        """Initialize all pipeline components."""
        
        # Trie for prefix matching
        if self.config.use_trie:
            self.trie = Trie(**self.config.trie_config)
            self.logger.info("Trie component initialized")
        else:
            self.trie = None
        
        # Sentence embedder for semantic search
        if self.config.use_semantic_search and self.config.embedder_config:
            self.embedder = SentenceEmbedder(self.config.embedder_config)
            self.embedder.eval()
            self.logger.info("Embedder component initialized")
        else:
            self.embedder = None
        
        # Vector indexer for fast similarity search
        if self.config.use_semantic_search and self.config.indexer_config:
            self.indexer = VectorIndexer(self.config.indexer_config)
            self.logger.info("Indexer component initialized")
        else:
            self.indexer = None
        
        # Ranking model for suggestion reordering
        if self.config.use_ranking and self.config.ranker_config:
            self.ranker = RankingModel(self.config.ranker_config)
            self.ranker.eval()
            self.logger.info("Ranker component initialized")
        else:
            self.ranker = None
        
        # Text generator for predictive suggestions
        if self.config.use_generation and self.config.generator_config:
            self.generator = ChatGenerator(self.config.generator_config)
            self.generator.eval()
            self.logger.info("Generator component initialized")
        else:
            self.generator = None
    
    async def suggest(
        self, 
        query: str, 
        user_id: Optional[str] = None,
        context: Optional[str] = None,
        max_suggestions: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Main suggestion method for backward compatibility.
        
        Args:
            query: Input query
            user_id: Optional user ID
            context: Optional context
            max_suggestions: Optional max suggestions limit
            
        Returns:
            List of suggestion dictionaries
        """
        request = AutocompleteRequest(
            query=query,
            user_id=user_id,
            context=context,
            max_suggestions=max_suggestions
        )
        
        response = await self.get_suggestions(request)
        
        # Convert to simple format
        return [
            {
                "text": suggestion.text,
                "score": suggestion.score,
                "source": suggestion.source,
                "confidence": suggestion.confidence
            }
            for suggestion in response.suggestions
        ]
    
    async def get_suggestions(self, request: AutocompleteRequest) -> AutocompleteResponse:
        """
        Get autocomplete suggestions for a query.
        
        Args:
            request: Autocomplete request
            
        Returns:
            Response with suggestions
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(request)
            cached_response = self.cache.get(cache_key)
            
            if cached_response:
                self.stats['requests_cached'] += 1
                cached_response.processing_time = time.time() - start_time
                return cached_response
            
            # Validate request
            if len(request.query.strip()) < self.config.min_query_length:
                return AutocompleteResponse(
                    suggestions=[],
                    query=request.query,
                    processing_time=time.time() - start_time,
                    request_id=request.request_id
                )
            
            # Generate suggestions from each component
            all_suggestions = []
            
            # Trie-based prefix matching
            if self.config.use_trie and self.trie:
                trie_suggestions = await self._get_trie_suggestions(request)
                all_suggestions.extend(trie_suggestions)
            
            # Semantic similarity search
            if self.config.use_semantic_search and self.embedder and self.indexer:
                semantic_suggestions = await self._get_semantic_suggestions(request)
                all_suggestions.extend(semantic_suggestions)
            
            # Predictive text generation
            if self.config.use_generation and self.generator:
                generation_suggestions = await self._get_generation_suggestions(request)
                all_suggestions.extend(generation_suggestions)
            
            # Rank and fuse suggestions
            if self.config.use_ranking and self.ranker and all_suggestions:
                ranked_suggestions = await self._rank_suggestions(request, all_suggestions)
            else:
                ranked_suggestions = await self._simple_fusion(all_suggestions)
            
            # Apply personalization
            if self.config.enable_personalization and request.user_id:
                ranked_suggestions = self._apply_personalization(
                    request.user_id, ranked_suggestions
                )
            
            # Limit results
            max_suggestions = request.max_suggestions or self.config.max_suggestions
            final_suggestions = ranked_suggestions[:max_suggestions]
            
            # Create response
            response = AutocompleteResponse(
                suggestions=final_suggestions,
                query=request.query,
                processing_time=time.time() - start_time,
                request_id=request.request_id,
                metadata={
                    'total_candidates': len(all_suggestions),
                    'sources_used': list(set(s.source for s in all_suggestions))
                }
            )
            
            # Cache response
            self.cache.put(cache_key, response)
            
            # Update statistics
            self.stats['requests_total'] += 1
            self._update_stats(response.processing_time)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in get_suggestions: {e}")
            self.stats['error_count'] += 1
            
            return AutocompleteResponse(
                suggestions=[],
                query=request.query,
                processing_time=time.time() - start_time,
                request_id=request.request_id,
                metadata={'error': str(e)}
            )
    
    async def _get_trie_suggestions(self, request: AutocompleteRequest) -> List[AutocompleteSuggestion]:
        """Get suggestions from Trie prefix matching."""
        if not self.trie:
            return []
        
        start_time = time.time()
        
        try:
            # Search trie for prefix matches
            results = self.trie.search_prefix(request.query.strip())
            
            suggestions = []
            for text, frequency in results:
                # Skip exact matches
                if text.lower() == request.query.lower():
                    continue
                
                suggestion = AutocompleteSuggestion(
                    text=text,
                    score=float(frequency),
                    source="trie",
                    confidence=min(1.0, frequency / 100.0),  # Normalize to [0, 1]
                    metadata={"frequency": frequency}
                )
                suggestions.append(suggestion)
            
            # Track timing
            self.stats['component_times']['trie'].append(time.time() - start_time)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in trie suggestions: {e}")
            return []
    
    async def _get_semantic_suggestions(self, request: AutocompleteRequest) -> List[AutocompleteSuggestion]:
        """Get suggestions from semantic similarity search."""
        if not self.embedder or not self.indexer:
            return []
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([request.query])[0]
            
            # Search vector index
            texts, similarities, metadata = self.indexer.search(
                query_embedding, 
                k=self.config.max_suggestions * 2
            )
            
            suggestions = []
            for text, similarity, meta in zip(texts, similarities, metadata):
                # Skip exact matches and low similarity
                if (text.lower() == request.query.lower() or 
                    similarity < 0.3):  # Threshold for relevance
                    continue
                
                suggestion = AutocompleteSuggestion(
                    text=text,
                    score=float(similarity),
                    source="semantic",
                    confidence=float(similarity),
                    metadata=meta
                )
                suggestions.append(suggestion)
            
            # Track timing
            self.stats['component_times']['semantic'].append(time.time() - start_time)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in semantic suggestions: {e}")
            return []
    
    async def _get_generation_suggestions(self, request: AutocompleteRequest) -> List[AutocompleteSuggestion]:
        """Get suggestions from text generation."""
        if not self.generator:
            return []
        
        start_time = time.time()
        
        try:
            # Tokenize input
            tokenizer = self.embedder.tokenizer if self.embedder else None
            if not tokenizer:
                return []
            
            input_ids = tokenizer.encode(
                request.query, 
                return_tensors='pt',
                add_special_tokens=True
            )
            
            # Generate completions
            with torch.no_grad():
                generated = self.generator.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + 20,
                    num_beams=3,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.generator.config.pad_token_id,
                    eos_token_id=self.generator.config.eos_token_id
                )
            
            # Decode generated text
            suggestions = []
            for seq in generated:
                # Skip the input part
                generated_part = seq[len(input_ids[0]):]
                if len(generated_part) > 0:
                    decoded = tokenizer.decode(generated_part, skip_special_tokens=True)
                    
                    # Create suggestion
                    full_text = request.query + decoded
                    suggestion = AutocompleteSuggestion(
                        text=full_text.strip(),
                        score=0.7,  # Default generation score
                        source="generation",
                        confidence=0.6,
                        metadata={"generated_tokens": len(generated_part)}
                    )
                    suggestions.append(suggestion)
            
            # Track timing
            self.stats['component_times']['generation'].append(time.time() - start_time)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in generation suggestions: {e}")
            return []
    
    async def _rank_suggestions(
        self, 
        request: AutocompleteRequest, 
        suggestions: List[AutocompleteSuggestion]
    ) -> List[AutocompleteSuggestion]:
        """Rank suggestions using the ranking model."""
        if not self.ranker or not suggestions:
            return suggestions
        
        try:
            # Extract features for ranking
            features_list = []
            for suggestion in suggestions:
                features = extract_ranking_features(
                    request.query,
                    suggestion.text,
                    context=request.context or "",
                    user_id=request.user_id,
                    metadata=suggestion.metadata
                )
                features_list.append(features)
            
            # Convert to tensor
            feature_tensor = torch.tensor(features_list, dtype=torch.float32)
            
            # Get ranking scores
            with torch.no_grad():
                ranking_scores = self.ranker(feature_tensor)
                ranking_scores = ranking_scores.squeeze().cpu().numpy()
            
            # Update suggestion scores
            for i, suggestion in enumerate(suggestions):
                # Combine original score with ranking score
                combined_score = (
                    suggestion.score * 0.3 + 
                    float(ranking_scores[i]) * 0.7
                )
                suggestion.score = combined_score
                suggestion.metadata['ranking_score'] = float(ranking_scores[i])
            
            # Sort by combined score
            suggestions.sort(key=lambda x: x.score, reverse=True)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error in ranking suggestions: {e}")
            return suggestions
    
    async def _simple_fusion(self, suggestions: List[AutocompleteSuggestion]) -> List[AutocompleteSuggestion]:
        """Simple fusion of suggestions without ranking model."""
        # Group by source
        by_source = defaultdict(list)
        for suggestion in suggestions:
            by_source[suggestion.source].append(suggestion)
        
        # Apply source weights
        for source, source_suggestions in by_source.items():
            weight = getattr(self.config, f"{source}_weight", 1.0)
            for suggestion in source_suggestions:
                suggestion.score *= weight
        
        # Combine and sort
        all_suggestions = []
        for source_suggestions in by_source.values():
            all_suggestions.extend(source_suggestions)
        
        # Remove duplicates
        seen = set()
        unique_suggestions = []
        for suggestion in all_suggestions:
            if suggestion.text not in seen:
                seen.add(suggestion.text)
                unique_suggestions.append(suggestion)
        
        # Sort by score
        unique_suggestions.sort(key=lambda x: x.score, reverse=True)
        
        return unique_suggestions
    
    def _apply_personalization(
        self, 
        user_id: str, 
        suggestions: List[AutocompleteSuggestion]
    ) -> List[AutocompleteSuggestion]:
        """Apply personalization based on user history."""
        if user_id not in self.user_histories:
            return suggestions
        
        user_history = self.user_histories[user_id]
        history_texts = set(user_history)
        
        # Boost suggestions that match user's history
        for suggestion in suggestions:
            # Check for partial matches in history
            boost = 0.0
            for history_text in history_texts:
                if (suggestion.text in history_text or 
                    history_text in suggestion.text):
                    boost += 0.1
            
            suggestion.score += boost
            suggestion.metadata['personalization_boost'] = boost
        
        # Re-sort with personalization
        suggestions.sort(key=lambda x: x.score, reverse=True)
        
        return suggestions
    
    def _get_cache_key(self, request: AutocompleteRequest) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.query,
            request.user_id or "anonymous",
            request.context or "",
            str(request.max_suggestions or self.config.max_suggestions)
        ]
        return "|".join(key_parts)
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        total_requests = self.stats['requests_total']
        current_avg = self.stats['avg_processing_time']
        
        self.stats['avg_processing_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.embedder:
            raise ValueError("Embedder not initialized")
        
        # Tokenize texts
        encoded = self.embedder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.embedder.config.max_length,
            return_tensors='pt'
        )
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.embedder.encode(
                encoded['input_ids'],
                encoded['attention_mask']
            )
        
        return embeddings.cpu().numpy()
    
    def record_user_action(self, user_id: str, selected_text: str):
        """Record user selection for personalization."""
        if self.config.enable_personalization and user_id:
            self.user_histories[user_id].append(selected_text)
            self.logger.debug(f"Recorded action for user {user_id}: {selected_text}")
    
    def add_training_data(
        self, 
        texts: List[str], 
        frequencies: Optional[List[int]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add training data to the pipeline components.
        
        Args:
            texts: List of text strings
            frequencies: Optional frequency counts for each text
            embeddings: Optional pre-computed embeddings
            metadata: Optional metadata for each text
        """
        
        # Add to Trie
        if self.trie:
            if frequencies:
                trie_data = list(zip(texts, frequencies))
            else:
                trie_data = [(text, 1) for text in texts]
            
            self.trie.bulk_insert(trie_data)
            self.logger.info(f"Added {len(texts)} texts to Trie")
        
        # Add to vector index
        if self.indexer and self.embedder:
            if embeddings is None:
                # Generate embeddings using the embedder
                embeddings = self._generate_embeddings(texts)
            
            self.indexer.add_vectors(embeddings, texts, metadata)
            self.logger.info(f"Added {len(texts)} vectors to index")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        stats = self.stats.copy()
        
        # Add component-specific stats
        if self.trie:
            stats['trie_stats'] = self.trie.get_statistics()
        
        if self.indexer:
            stats['indexer_stats'] = self.indexer.get_statistics()
        
        # Add cache stats
        stats['cache_size'] = len(self.cache.cache)
        stats['cache_hit_rate'] = (
            self.stats['requests_cached'] / max(self.stats['requests_total'], 1)
        )
        
        return stats


# TODO: Implement the following enhancements:
# 1. Distributed pipeline with microservices architecture
# 2. Real-time model updates and A/B testing framework
# 3. Advanced caching strategies (Redis, Memcached)
# 4. Monitoring and alerting for production deployment
# 5. Auto-scaling based on load and performance metrics
# 6. Privacy-preserving personalization with federated learning
# 7. Multi-language support with language detection
# 8. Integration with external knowledge bases and APIs
