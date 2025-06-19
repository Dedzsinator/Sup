"""
Unit tests for the Trie implementation.

This module provides comprehensive test coverage for the Trie class,
including edge cases, performance tests, and thread safety validation.
"""

import unittest
import threading
import time
from typing import List, Tuple
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from trie import Trie, TrieNode


class TestTrieNode(unittest.TestCase):
    """Test cases for TrieNode class."""
    
    def test_node_initialization(self):
        """Test TrieNode default initialization."""
        node = TrieNode()
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.is_end_word)
        self.assertEqual(node.frequency, 0)
        self.assertIsNone(node.word)
    
    def test_node_with_values(self):
        """Test TrieNode initialization with custom values."""
        node = TrieNode(is_end_word=True, frequency=5, word="test")
        self.assertTrue(node.is_end_word)
        self.assertEqual(node.frequency, 5)
        self.assertEqual(node.word, "test")


class TestTrie(unittest.TestCase):
    """Test cases for Trie class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.trie = Trie()
        self.sample_data = [
            ("hello", 10),
            ("hello world", 8),
            ("hello there", 5),
            ("help", 3),
            ("helicopter", 2),
            ("hell", 1)
        ]
    
    def test_trie_initialization(self):
        """Test Trie default initialization."""
        trie = Trie()
        self.assertFalse(trie.case_sensitive)
        self.assertEqual(trie.max_suggestions, 10)
        self.assertIsInstance(trie.root, TrieNode)
        
        # Test custom initialization
        custom_trie = Trie(case_sensitive=True, max_suggestions=5)
        self.assertTrue(custom_trie.case_sensitive)
        self.assertEqual(custom_trie.max_suggestions, 5)
    
    def test_insert_single_word(self):
        """Test inserting a single word."""
        self.trie.insert("hello", 5)
        self.assertTrue(self.trie.search_exact("hello"))
        self.assertEqual(len(self.trie.search_prefix("hello")), 1)
        self.assertEqual(self.trie.search_prefix("hello")[0], ("hello", 5))
    
    def test_insert_multiple_words(self):
        """Test inserting multiple words."""
        for word, freq in self.sample_data:
            self.trie.insert(word, freq)
        
        # Verify all words exist
        for word, _ in self.sample_data:
            self.assertTrue(self.trie.search_exact(word))
        
        # Test statistics
        stats = self.trie.get_statistics()
        self.assertEqual(stats['word_count'], len(self.sample_data))
    
    def test_bulk_insert(self):
        """Test bulk insertion of words."""
        self.trie.bulk_insert(self.sample_data)
        
        for word, _ in self.sample_data:
            self.assertTrue(self.trie.search_exact(word))
    
    def test_prefix_search(self):
        """Test prefix search functionality."""
        self.trie.bulk_insert(self.sample_data)
        
        # Search for "hel" prefix
        results = self.trie.search_prefix("hel")
        expected_words = {"hello", "hello world", "hello there", "help", "helicopter", "hell"}
        result_words = {word for word, _ in results}
        self.assertEqual(result_words, expected_words)
        
        # Verify sorting by frequency (descending)
        self.assertEqual(results[0][0], "hello")  # Highest frequency
        self.assertEqual(results[0][1], 10)
    
    def test_prefix_search_ordering(self):
        """Test that prefix search results are ordered by frequency."""
        self.trie.bulk_insert(self.sample_data)
        
        results = self.trie.search_prefix("hello")
        frequencies = [freq for _, freq in results]
        
        # Should be in descending order
        self.assertEqual(frequencies, sorted(frequencies, reverse=True))
    
    def test_case_sensitivity(self):
        """Test case sensitivity settings."""
        # Case insensitive (default)
        case_insensitive_trie = Trie(case_sensitive=False)
        case_insensitive_trie.insert("Hello", 5)
        self.assertTrue(case_insensitive_trie.search_exact("hello"))
        self.assertTrue(case_insensitive_trie.search_exact("HELLO"))
        
        # Case sensitive
        case_sensitive_trie = Trie(case_sensitive=True)
        case_sensitive_trie.insert("Hello", 5)
        self.assertTrue(case_sensitive_trie.search_exact("Hello"))
        self.assertFalse(case_sensitive_trie.search_exact("hello"))
    
    def test_max_suggestions_limit(self):
        """Test max suggestions limit functionality."""
        trie = Trie(max_suggestions=3)
        
        # Insert more words than the limit
        words = [f"test{i}" for i in range(10)]
        for i, word in enumerate(words):
            trie.insert(word, i)
        
        results = trie.search_prefix("test")
        self.assertLessEqual(len(results), 3)
    
    def test_update_frequency(self):
        """Test frequency update functionality."""
        self.trie.insert("hello", 5)
        self.assertTrue(self.trie.update_frequency("hello", 15))
        
        result = self.trie.search_prefix("hello")[0]
        self.assertEqual(result[1], 15)
        
        # Test updating non-existent word
        self.assertFalse(self.trie.update_frequency("nonexistent", 10))
    
    def test_remove_word(self):
        """Test word removal functionality."""
        self.trie.bulk_insert(self.sample_data)
        
        # Remove a word
        self.assertTrue(self.trie.remove("hello"))
        self.assertFalse(self.trie.search_exact("hello"))
        
        # Verify other words still exist
        self.assertTrue(self.trie.search_exact("hello world"))
        
        # Test removing non-existent word
        self.assertFalse(self.trie.remove("nonexistent"))
    
    def test_get_top_k_by_frequency(self):
        """Test getting top k words by frequency."""
        self.trie.bulk_insert(self.sample_data)
        
        top_3 = self.trie.get_top_k_by_frequency(3)
        self.assertEqual(len(top_3), 3)
        
        # Should be sorted by frequency
        frequencies = [freq for _, freq in top_3]
        self.assertEqual(frequencies, sorted(frequencies, reverse=True))
        
        # Top word should be "hello" with frequency 10
        self.assertEqual(top_3[0], ("hello", 10))
    
    def test_empty_and_whitespace_inputs(self):
        """Test handling of empty and whitespace inputs."""
        # Empty string
        self.trie.insert("", 5)
        self.assertEqual(len(self.trie.search_prefix("")), 0)
        
        # Whitespace only
        self.trie.insert("   ", 5)
        self.assertEqual(len(self.trie.search_prefix("   ")), 0)
        
        # Mixed whitespace
        self.trie.insert("  hello  ", 5)
        self.assertTrue(self.trie.search_exact("hello"))
    
    def test_clear_trie(self):
        """Test clearing the Trie."""
        self.trie.bulk_insert(self.sample_data)
        self.assertGreater(self.trie.get_statistics()['word_count'], 0)
        
        self.trie.clear()
        self.assertEqual(self.trie.get_statistics()['word_count'], 0)
        self.assertEqual(len(self.trie.search_prefix("hel")), 0)
    
    def test_statistics(self):
        """Test statistics collection."""
        self.trie.bulk_insert(self.sample_data)
        
        stats = self.trie.get_statistics()
        self.assertIn('word_count', stats)
        self.assertIn('node_count', stats)
        self.assertIn('max_suggestions', stats)
        self.assertIn('case_sensitive', stats)
        
        self.assertEqual(stats['word_count'], len(self.sample_data))
        self.assertGreater(stats['node_count'], 0)


class TestTrieThreadSafety(unittest.TestCase):
    """Test thread safety of Trie operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trie = Trie()
        self.results = []
        self.errors = []
    
    def worker_insert(self, words: List[Tuple[str, int]]):
        """Worker function for concurrent insertions."""
        try:
            for word, freq in words:
                self.trie.insert(f"worker_{word}", freq)
        except Exception as e:
            self.errors.append(e)
    
    def worker_search(self, prefixes: List[str]):
        """Worker function for concurrent searches."""
        try:
            for prefix in prefixes:
                results = self.trie.search_prefix(prefix)
                self.results.extend(results)
        except Exception as e:
            self.errors.append(e)
    
    def test_concurrent_insertions(self):
        """Test concurrent insertions from multiple threads."""
        words_per_thread = [
            [("hello1", 1), ("world1", 2)],
            [("hello2", 3), ("world2", 4)],
            [("hello3", 5), ("world3", 6)]
        ]
        
        threads = []
        for words in words_per_thread:
            thread = threading.Thread(target=self.worker_insert, args=(words,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check that no errors occurred
        self.assertEqual(len(self.errors), 0)
        
        # Verify all words were inserted
        for words in words_per_thread:
            for word, _ in words:
                self.assertTrue(self.trie.search_exact(f"worker_{word}"))
    
    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        # Pre-populate with some data
        initial_data = [(f"initial_{i}", i) for i in range(100)]
        self.trie.bulk_insert(initial_data)
        
        # Start writer threads
        write_threads = []
        for i in range(3):
            words = [(f"concurrent_{i}_{j}", j) for j in range(10)]
            thread = threading.Thread(target=self.worker_insert, args=(words,))
            write_threads.append(thread)
            thread.start()
        
        # Start reader threads
        read_threads = []
        for i in range(3):
            prefixes = ["initial", "concurrent"]
            thread = threading.Thread(target=self.worker_search, args=(prefixes,))
            read_threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in write_threads + read_threads:
            thread.join()
        
        # Check that no errors occurred
        self.assertEqual(len(self.errors), 0)


class TestTriePerformance(unittest.TestCase):
    """Performance tests for Trie operations."""
    
    def test_insertion_performance(self):
        """Test insertion performance with large datasets."""
        trie = Trie()
        
        # Generate test data
        words = [f"performance_test_word_{i}" for i in range(10000)]
        
        start_time = time.time()
        for word in words:
            trie.insert(word, 1)
        insertion_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(insertion_time, 5.0, "Insertion took too long")
        
        # Verify correctness
        self.assertEqual(trie.get_statistics()['word_count'], len(words))
    
    def test_search_performance(self):
        """Test search performance with large datasets."""
        trie = Trie()
        
        # Insert test data
        words = [f"search_test_{i}" for i in range(5000)]
        trie.bulk_insert([(word, 1) for word in words])
        
        # Test search performance
        start_time = time.time()
        for i in range(1000):
            results = trie.search_prefix("search_test")
            self.assertGreater(len(results), 0)
        search_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(search_time, 2.0, "Search took too long")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
