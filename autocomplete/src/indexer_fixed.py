"""
FAISS-based Vector Indexer for Fast Semantic Search

This module provides a wrapper around FAISS for efficient approximate nearest
neighbor (ANN) search in the embedding space. Supports multiple index types
and real-time updates for chat autocomplete.

Features:
- Multiple FAISS index types (Flat, IVF, HNSW)
- GPU acceleration support
- Incremental index updates
- Persistence and loading
- Batch processing for efficiency
- Similarity search with metadata

Author: Generated for Sup Chat Application
"""

import faiss
import numpy as np
import pickle
import json
import logging
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import time
from collections import defaultdict


@dataclass
class IndexConfig:
    """Configuration for the FAISS indexer."""
    index_type: str = "Flat"  # Flat, IVF100, IVF1000, HNSW32
    embedding_dim: int = 384
    metric: str = "IP"  # IP (inner product), L2 (euclidean)
    use_gpu: bool = False
    gpu_device: int = 0
    nprobe: int = 10  # Number of probes for IVF indexes
    m: int = 32  # Number of connections for HNSW
    ef_construction: int = 200  # Construction parameter for HNSW
    ef_search: int = 64  # Search parameter for HNSW
    normalize_vectors: bool = True
    batch_size: int = 1000


class VectorIndexer:
    """
    High-performance vector indexer using FAISS for semantic search.
    
    This class provides fast approximate nearest neighbor search for
    sentence embeddings in real-time chat autocomplete scenarios.
    """
    
    def __init__(self, config: IndexConfig):
        """
        Initialize the vector indexer.
        
        Args:
            config: Configuration object with indexer parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize FAISS index
        self.index = self._create_index()
        self.gpu_index = None
        
        # Metadata storage
        self.id_to_text: Dict[int, str] = {}
        self.text_to_id: Dict[str, int] = {}
        self.id_to_metadata: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_vectors': 0,
            'search_count': 0,
            'avg_search_time': 0.0,
            'last_update': None
        }
        
        self.logger.info(f"Initialized {config.index_type} index with dimension {config.embedding_dim}")
    
    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on configuration.
        
        Returns:
            Initialized FAISS index
        """
        dim = self.config.embedding_dim
        
        if self.config.index_type == "Flat":
            if self.config.metric == "IP":
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexFlatL2(dim)
                
        elif self.config.index_type.startswith("IVF"):
            nlist = int(self.config.index_type[3:])  # Extract number from IVFxxx
            
            if self.config.metric == "IP":
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
                
            index.nprobe = self.config.nprobe
            
        elif self.config.index_type.startswith("HNSW"):
            m = int(self.config.index_type[4:])  # Extract M from HNSWxx
            
            if self.config.metric == "IP":
                index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            else:
                index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
                
            index.hnsw.ef_construction = self.config.ef_construction
            index.hnsw.ef_search = self.config.ef_search
            
        else:
            raise ValueError(f"Unsupported index type: {self.config.index_type}")
        
        # Move to GPU if requested and available
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, self.config.gpu_device, index)
            self.logger.info(f"Moved index to GPU {self.config.gpu_device}")
        
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors if configured.
        
        Args:
            vectors: Input vectors [n_vectors, dim]
            
        Returns:
            Normalized vectors
        """
        if self.config.normalize_vectors:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add vectors to the index with associated texts and metadata.
        
        Args:
            vectors: Embedding vectors [n_vectors, embedding_dim]
            texts: Associated text strings
            metadata: Optional metadata for each vector
            
        Returns:
            List of assigned IDs for the vectors
        """
        if len(vectors) != len(texts):
            raise ValueError("Number of vectors must match number of texts")
        
        with self._lock:
            # Normalize vectors if needed
            vectors = self._normalize_vectors(vectors.astype(np.float32))
            
            # Assign IDs and store metadata
            ids = []
            for i, text in enumerate(texts):
                vector_id = self.next_id
                ids.append(vector_id)
                
                self.id_to_text[vector_id] = text
                self.text_to_id[text] = vector_id
                
                if metadata:
                    self.id_to_metadata[vector_id] = metadata[i] if i < len(metadata) else {}
                else:
                    self.id_to_metadata[vector_id] = {}
                
                self.next_id += 1
            
            # Add to FAISS index
            self.index.add(vectors)
            
            # Update statistics
            self.stats['total_vectors'] += len(vectors)
            self.stats['last_update'] = time.time()
            
            self.logger.info(f"Added {len(vectors)} vectors to index")
            return ids
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_vector: Query embedding vector [embedding_dim]
            k: Number of nearest neighbors to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Tuple of (texts, similarities, metadata)
        """
        start_time = time.time()
        
        with self._lock:
            if self.index.ntotal == 0:
                return [], [], []
            
            # Normalize query vector
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            query_vector = self._normalize_vectors(query_vector.astype(np.float32))
            
            # Perform search
            similarities, indices = self.index.search(query_vector, k)
            
            # Convert results
            texts = []
            metadata_list = []
            valid_similarities = []
            
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                # Get text and metadata
                text = self.id_to_text.get(idx, "")
                metadata = self.id_to_metadata.get(idx, {})
                
                # Skip removed vectors
                if metadata.get('_removed', False):
                    continue
                
                # Apply metadata filters if specified
                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                
                texts.append(text)
                metadata_list.append(metadata)
                valid_similarities.append(float(similarity))
            
            # Update statistics
            search_time = time.time() - start_time
            self.stats['search_count'] += 1
            self.stats['avg_search_time'] = (
                (self.stats['avg_search_time'] * (self.stats['search_count'] - 1) + search_time) /
                self.stats['search_count']
            )
            
            return texts, valid_similarities, metadata_list
    
    def batch_search(
        self, 
        query_vectors: np.ndarray, 
        k: int = 10
    ) -> List[Tuple[List[str], List[float], List[Dict[str, Any]]]]:
        """
        Perform batch search for multiple query vectors.
        
        Args:
            query_vectors: Query embedding vectors [n_queries, embedding_dim]
            k: Number of nearest neighbors to return per query
            
        Returns:
            List of search results for each query
        """
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = self.config.batch_size
        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:i + batch_size]
            
            for query_vector in batch:
                result = self.search(query_vector, k)
                results.append(result)
        
        return results
    
    def update_vector(self, vector_id: int, new_vector: np.ndarray) -> bool:
        """
        Update an existing vector in the index.
        
        Note: FAISS doesn't support in-place updates, so this requires
        rebuilding the index or using a more complex update strategy.
        
        Args:
            vector_id: ID of the vector to update
            new_vector: New embedding vector
            
        Returns:
            True if update was successful
        """
        # For now, we don't support direct updates
        # In production, consider using multiple indexes or rebuild strategies
        self.logger.warning("Vector updates not supported in current implementation")
        return False
    
    def remove_vector(self, vector_id: int) -> bool:
        """
        Remove a vector from the index.
        
        Note: FAISS doesn't support efficient removal, so this marks
        the vector as removed in metadata only.
        
        Args:
            vector_id: ID of the vector to remove
            
        Returns:
            True if removal was successful
        """
        with self._lock:
            if vector_id in self.id_to_text:
                text = self.id_to_text[vector_id]
                
                # Mark as removed
                self.id_to_metadata[vector_id]['_removed'] = True
                
                # Clean up mappings
                del self.id_to_text[vector_id]
                if text in self.text_to_id:
                    del self.text_to_id[text]
                
                self.logger.info(f"Marked vector {vector_id} as removed")
                return True
            
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        with self._lock:
            active_vectors = len([
                vid for vid, metadata in self.id_to_metadata.items()
                if not metadata.get('_removed', False)
            ])
            
            return {
                **self.stats,
                'active_vectors': active_vectors,
                'removed_vectors': self.stats['total_vectors'] - active_vectors,
                'index_type': self.config.index_type,
                'embedding_dim': self.config.embedding_dim
            }
    
    def save(self, index_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None):
        """
        Save the index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Optional path to save metadata (defaults to index_path with .meta extension)
        """
        index_path = Path(index_path)
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.meta')
        else:
            metadata_path = Path(metadata_path)
        
        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save FAISS index
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                # Move to CPU before saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata = {
                'id_to_text': self.id_to_text,
                'text_to_id': self.text_to_id,
                'id_to_metadata': self.id_to_metadata,
                'next_id': self.next_id,
                'stats': self.stats,
                'config': asdict(self.config)
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        self.logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")
    
    def load(self, index_path: Union[str, Path], metadata_path: Optional[Union[str, Path]] = None):
        """
        Load the index and metadata from disk.
        
        Args:
            index_path: Path to the FAISS index file
            metadata_path: Optional path to metadata file
        """
        index_path = Path(index_path)
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.meta')
        else:
            metadata_path = Path(metadata_path)
        
        with self._lock:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Move to GPU if configured
            if self.config.use_gpu and faiss.get_num_gpus() > 0:
                gpu_resource = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resource, self.config.gpu_device, self.index)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.id_to_text = metadata['id_to_text']
            self.text_to_id = metadata['text_to_id']
            self.id_to_metadata = metadata['id_to_metadata']
            self.next_id = metadata['next_id']
            self.stats = metadata['stats']
        
        self.logger.info(f"Loaded index from {index_path} with {self.index.ntotal} vectors")
    
    def clear(self):
        """Clear all data from the indexer."""
        with self._lock:
            # Recreate empty index
            self.index = self._create_index()
            
            # Clear metadata
            self.id_to_text.clear()
            self.text_to_id.clear()
            self.id_to_metadata.clear()
            self.next_id = 0
            
            # Reset statistics
            self.stats = {
                'total_vectors': 0,
                'search_count': 0,
                'avg_search_time': 0.0,
                'last_update': None
            }
        
        self.logger.info("Cleared all data from indexer")


class HybridIndexer:
    """
    Hybrid indexer that combines multiple FAISS indexes for different use cases.
    
    For example:
    - Fast flat index for recent messages
    - Larger IVF index for historical messages
    - Separate indexes for different message types
    """
    
    def __init__(self, configs: Dict[str, IndexConfig]):
        """
        Initialize hybrid indexer with multiple index configurations.
        
        Args:
            configs: Dictionary mapping index names to configurations
        """
        self.indexers = {}
        self.logger = logging.getLogger(__name__)
        
        for name, config in configs.items():
            self.indexers[name] = VectorIndexer(config)
            self.logger.info(f"Created indexer '{name}' with type {config.index_type}")
    
    def add_vectors(
        self, 
        index_name: str,
        vectors: np.ndarray, 
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """Add vectors to a specific index."""
        if index_name not in self.indexers:
            raise ValueError(f"Index '{index_name}' not found")
        
        return self.indexers[index_name].add_vectors(vectors, texts, metadata)
    
    def search_all(
        self, 
        query_vector: np.ndarray, 
        k: int = 10,
        merge_strategy: str = "score"
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Search across all indexes and merge results.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            merge_strategy: How to merge results ("score", "round_robin")
            
        Returns:
            Merged search results
        """
        all_results = []
        
        # Search each index
        for name, indexer in self.indexers.items():
            texts, similarities, metadata = indexer.search(query_vector, k)
            
            # Add index name to metadata
            for i, meta in enumerate(metadata):
                meta['_source_index'] = name
            
            all_results.extend(list(zip(texts, similarities, metadata)))
        
        # Sort by similarity and take top k
        if merge_strategy == "score":
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_results = all_results[:k]
        else:
            # Simple round-robin merge
            top_results = all_results[:k]
        
        # Unpack results
        texts, similarities, metadata = zip(*top_results) if top_results else ([], [], [])
        
        return list(texts), list(similarities), list(metadata)


# TODO: Implement the following enhancements:
# 1. Dynamic index selection based on query characteristics
# 2. Incremental index updates with efficient rebuilding
# 3. Distributed indexing across multiple machines
# 4. Memory-mapped indexes for large-scale deployment
# 5. Quantization for memory efficiency (PQ, OPQ, etc.)
# 6. Index warming strategies for cold start scenarios
# 7. Real-time index monitoring and alerting
# 8. Automatic parameter tuning based on data characteristics
