#!/usr/bin/env python3
"""
Test script to verify all autocomplete components are fully implemented.

This script tests:
1. Trie implementation and all methods
2. Embedder functionality 
3. Vector indexer operations
4. Ranking model features
5. Text generator capabilities
6. Full pipeline integration

Author: Generated for Sup Chat Application
"""

import sys
import traceback
import logging
import torch
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trie_implementation():
    """Test Trie functionality."""
    logger.info("Testing Trie implementation...")
    
    try:
        from trie import Trie
        
        # Initialize trie
        trie = Trie(case_sensitive=False, max_suggestions=10)
        
        # Test basic operations
        test_words = [
            ("hello world", 10),
            ("hello there", 8),
            ("hello everyone", 5),
            ("hi there", 15),
            ("good morning", 12)
        ]
        
        # Test bulk insert
        trie.bulk_insert(test_words)
        logger.info(f"Trie contains {len(trie)} words")
        
        # Test search operations
        exact_search = trie.search_exact("hello world")
        logger.info(f"Exact search for 'hello world': {exact_search}")
        
        prefix_results = trie.search_prefix("hello")
        logger.info(f"Prefix search for 'hello': {prefix_results}")
        
        # Test statistics
        stats = trie.get_statistics()
        logger.info(f"Trie statistics: {stats}")
        
        # Test save/load functionality
        trie.save("test_trie.pkl")
        
        new_trie = Trie()
        new_trie.load("test_trie.pkl")
        logger.info(f"Loaded trie contains {len(new_trie)} words")
        
        # Test export to JSON
        trie.export_to_json("test_trie.json")
        
        logger.info("✓ Trie implementation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Trie implementation test failed: {e}")
        traceback.print_exc()
        return False

def test_embedder_implementation():
    """Test Embedder functionality."""
    logger.info("Testing Embedder implementation...")
    
    try:
        from embedder import SentenceEmbedder, EmbedderConfig, ContrastiveDataset, ContrastiveLoss
        
        # Create config
        config = EmbedderConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
            max_length=128,
            batch_size=4
        )
        
        # Initialize embedder
        embedder = SentenceEmbedder(config)
        logger.info("✓ Embedder initialized")
        
        # Test encoding
        test_texts = ["Hello world", "How are you?", "Good morning"]
        
        # Tokenize
        encoded = embedder.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors='pt'
        )
        
        # Test encoding
        embeddings = embedder.encode(encoded['input_ids'], encoded['attention_mask'])
        logger.info(f"✓ Generated embeddings shape: {embeddings.shape}")
        
        # Test contrastive dataset
        pos_pairs = [("hello", "hi"), ("good morning", "good day")]
        neg_pairs = [("hello", "goodbye"), ("morning", "night")]
        
        dataset = ContrastiveDataset(pos_pairs, neg_pairs, embedder.tokenizer)
        logger.info(f"✓ Contrastive dataset created with {len(dataset)} pairs")
        
        # Test contrastive loss
        loss_fn = ContrastiveLoss()
        emb1 = torch.randn(2, 384)
        emb2 = torch.randn(2, 384)
        labels = torch.tensor([1.0, 0.0])
        
        loss = loss_fn(emb1, emb2, labels)
        logger.info(f"✓ Contrastive loss computed: {loss.item()}")
        
        logger.info("✓ Embedder implementation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Embedder implementation test failed: {e}")
        traceback.print_exc()
        return False

def test_indexer_implementation():
    """Test Vector Indexer functionality."""
    logger.info("Testing Vector Indexer implementation...")
    
    try:
        from indexer import VectorIndexer, IndexConfig, HybridIndexer
        
        # Create config
        config = IndexConfig(
            index_type="Flat",
            embedding_dim=384,
            use_gpu=False
        )
        
        # Initialize indexer
        indexer = VectorIndexer(config)
        logger.info("✓ Vector indexer initialized")
        
        # Test adding vectors
        test_vectors = np.random.randn(10, 384).astype(np.float32)
        test_texts = [f"Document {i}" for i in range(10)]
        test_metadata = [{"id": i, "type": "test"} for i in range(10)]
        
        ids = indexer.add_vectors(test_vectors, test_texts, test_metadata)
        logger.info(f"✓ Added {len(ids)} vectors to index")
        
        # Test search
        query_vector = np.random.randn(384).astype(np.float32)
        texts, similarities, metadata = indexer.search(query_vector, k=5)
        logger.info(f"✓ Search returned {len(texts)} results")
        
        # Test batch search
        query_vectors = np.random.randn(3, 384).astype(np.float32)
        batch_results = indexer.batch_search(query_vectors, k=3)
        logger.info(f"✓ Batch search returned {len(batch_results)} result sets")
        
        # Test statistics
        stats = indexer.get_statistics()
        logger.info(f"✓ Indexer statistics: {stats}")
        
        # Test hybrid indexer
        configs = {
            "fast": IndexConfig(index_type="Flat", embedding_dim=384),
            "large": IndexConfig(index_type="IVF100", embedding_dim=384)
        }
        
        hybrid_indexer = HybridIndexer(configs)
        logger.info("✓ Hybrid indexer initialized")
        
        # Test hybrid operations
        hybrid_indexer.add_vectors("fast", test_vectors[:5], test_texts[:5])
        hybrid_indexer.add_vectors("large", test_vectors[5:], test_texts[5:])
        
        hybrid_results = hybrid_indexer.search_all(query_vector, k=5)
        logger.info(f"✓ Hybrid search returned {len(hybrid_results[0])} results")
        
        logger.info("✓ Vector Indexer implementation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector Indexer implementation test failed: {e}")
        traceback.print_exc()
        return False

def test_ranker_implementation():
    """Test Ranking Model functionality."""
    logger.info("Testing Ranking Model implementation...")
    
    try:
        from ranker import RankingModel, RankerConfig, extract_ranking_features, RankingDataset
        
        # Create config
        config = RankerConfig(
            query_embedding_dim=384,
            candidate_embedding_dim=384,
            hidden_dim=256,
            learning_rate=1e-4
        )
        
        # Initialize ranker
        ranker = RankingModel(config)
        logger.info("✓ Ranking model initialized")
        
        # Test feature extraction
        features = extract_ranking_features(
            query="hello world",
            candidate="hello there world",
            context="greeting context",
            user_id="user123"
        )
        logger.info(f"✓ Extracted {len(features)} features")
        
        # Test forward pass
        query_emb = torch.randn(5, config.query_embedding_dim)
        candidate_emb = torch.randn(5, config.candidate_embedding_dim)
        features = torch.randn(5, config.feature_dim)
        
        scores = ranker(query_emb, candidate_emb, features)
        logger.info(f"✓ Ranking scores shape: {scores.shape}")
        
        # Test dataset
        queries = ["hello", "good morning", "how are you"]
        candidates_list = [
            ["hello world", "hello there"],
            ["good morning everyone", "morning"],
            ["how are you doing", "how's it going"]
        ]
        scores_list = [
            [1.0, 0.8],
            [1.0, 0.6],
            [1.0, 0.9]
        ]
        
        # Create dummy embeddings
        query_embeddings = np.random.randn(3, config.query_embedding_dim)
        candidate_embeddings = [
            np.random.randn(2, config.candidate_embedding_dim),
            np.random.randn(2, config.candidate_embedding_dim),
            np.random.randn(2, config.candidate_embedding_dim)
        ]
        
        dataset = RankingDataset(
            queries, 
            candidates_list, 
            scores_list,
            query_embeddings,
            candidate_embeddings
        )
        logger.info(f"✓ Ranking dataset created with {len(dataset)} samples")
        
        logger.info("✓ Ranking Model implementation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ranking Model implementation test failed: {e}")
        traceback.print_exc()
        return False

def test_generator_implementation():
    """Test Text Generator functionality."""
    logger.info("Testing Text Generator implementation...")
    
    try:
        from generator import ChatGenerator, GeneratorConfig, ChatDataset, GeneratorTrainer
        from transformers import AutoTokenizer
        
        # Create config
        config = GeneratorConfig(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=4,
            max_seq_len=128
        )
        
        # Initialize generator
        generator = ChatGenerator(config)
        logger.info("✓ Text generator initialized")
        
        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 50))
        logits, _ = generator(input_ids)
        logger.info(f"✓ Generator output shape: {logits.shape}")
        
        # Test generation
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        generated = generator.generate(
            prompt, 
            max_length=20,
            temperature=0.8,
            do_sample=True
        )
        logger.info(f"✓ Generated sequence shape: {generated.shape}")
        
        logger.info("✓ Text Generator implementation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Text Generator implementation test failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline_integration():
    """Test full pipeline integration."""
    logger.info("Testing Pipeline integration...")
    
    try:
        from pipeline import AutocompletePipeline, PipelineConfig, AutocompleteRequest
        from embedder import EmbedderConfig
        from indexer import IndexConfig
        from ranker import RankerConfig
        from generator import GeneratorConfig
        
        # Create minimal config for testing
        config = PipelineConfig(
            use_trie=True,
            use_semantic_search=False,  # Skip for this test to avoid model loading
            use_ranking=False,
            use_generation=False,
            max_suggestions=5
        )
        
        # Initialize pipeline
        pipeline = AutocompletePipeline(config)
        logger.info("✓ Pipeline initialized")
        
        # Add some test data
        test_texts = [
            "hello world",
            "hello there", 
            "good morning",
            "good afternoon",
            "how are you"
        ]
        frequencies = [10, 8, 15, 12, 20]
        
        pipeline.add_training_data(test_texts, frequencies)
        logger.info("✓ Added training data to pipeline")
        
        # Test suggestion generation
        request = AutocompleteRequest(
            query="hello",
            user_id="test_user"
        )
        
        import asyncio
        async def test_suggestions():
            response = await pipeline.get_suggestions(request)
            return response
        
        # Run async test
        response = asyncio.run(test_suggestions())
        logger.info(f"✓ Generated {len(response.suggestions)} suggestions")
        
        for suggestion in response.suggestions:
            logger.info(f"  - {suggestion.text} (score: {suggestion.score:.3f}, source: {suggestion.source})")
        
        # Test statistics
        stats = pipeline.get_statistics()
        logger.info(f"✓ Pipeline statistics: {stats}")
        
        logger.info("✓ Pipeline integration tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Pipeline integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all implementation tests."""
    logger.info("Starting comprehensive implementation tests...")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Trie", test_trie_implementation),
        ("Embedder", test_embedder_implementation), 
        ("Indexer", test_indexer_implementation),
        ("Ranker", test_ranker_implementation),
        ("Generator", test_generator_implementation),
        ("Pipeline", test_pipeline_integration)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} tests...")
        logger.info('='*50)
        
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:15} : {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL IMPLEMENTATIONS COMPLETE! 🎉")
        return 0
    else:
        logger.error(f"❌ {total - passed} implementations still need work")
        return 1

if __name__ == "__main__":
    sys.exit(main())
