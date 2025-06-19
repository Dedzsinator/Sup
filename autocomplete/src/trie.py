"""
Trie (Prefix Tree) Implementation for Fast Symbolic Prefix Search

This module provides a memory-efficient Trie data structure optimized for
real-time autocomplete suggestions in chat applications.

Features:
- Fast O(m) prefix search where m is prefix length
- Memory-efficient node structure
- Support for frequency-based ranking
- Configurable max suggestions per prefix
- Thread-safe operations for concurrent access

Author: Generated for Sup Chat Application
"""

import threading
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class TrieNode:
    """
    Represents a single node in the Trie structure.
    
    Attributes:
        children: Dictionary mapping characters to child nodes
        is_end_word: Boolean indicating if this node marks the end of a word
        frequency: How often this word/phrase has been used (for ranking)
        word: The complete word if this is an end node (for efficient retrieval)
    """
    children: Dict[str, 'TrieNode'] = field(default_factory=dict)
    is_end_word: bool = False
    frequency: int = 0
    word: Optional[str] = None


class Trie:
    """
    High-performance Trie implementation for autocomplete suggestions.
    
    This Trie supports:
    - Case-insensitive search (configurable)
    - Frequency-based ranking of suggestions
    - Bulk insertion for efficient initialization
    - Thread-safe operations
    - Memory-efficient storage
    
    Example:
        >>> trie = Trie()
        >>> trie.insert("hello world", frequency=5)
        >>> trie.insert("hello there", frequency=3)
        >>> suggestions = trie.search_prefix("hello")
        >>> print(suggestions)  # [("hello world", 5), ("hello there", 3)]
    """
    
    def __init__(self, case_sensitive: bool = False, max_suggestions: int = 10):
        """
        Initialize the Trie.
        
        Args:
            case_sensitive: If False, converts all text to lowercase
            max_suggestions: Maximum number of suggestions to return per prefix
        """
        self.root = TrieNode()
        self.case_sensitive = case_sensitive
        self.max_suggestions = max_suggestions
        self._lock = threading.RLock()  # For thread safety
        self._word_count = 0
        
    def __len__(self) -> int:
        """Return the number of words stored in the Trie."""
        return self._word_count
        
    def _normalize_text(self, text: str) -> str:
        """Normalize text based on case sensitivity setting."""
        return text if self.case_sensitive else text.lower()
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert a word into the Trie with optional frequency.
        
        Args:
            word: The word/phrase to insert
            frequency: Usage frequency for ranking (default: 1)
            
        Time Complexity: O(m) where m is the length of the word
        """
        if not word.strip():
            return
            
        with self._lock:
            word = self._normalize_text(word.strip())
            current = self.root
            
            # Traverse or create path for each character
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            
            # Mark end of word and update frequency
            if not current.is_end_word:
                self._word_count += 1
            current.is_end_word = True
            current.frequency = max(current.frequency, frequency)
            current.word = word
    
    def bulk_insert(self, words: List[Tuple[str, int]]) -> None:
        """
        Efficiently insert multiple words with their frequencies.
        
        Args:
            words: List of (word, frequency) tuples
            
        This method is more efficient than multiple insert() calls
        as it reduces lock overhead.
        """
        with self._lock:
            for word, frequency in words:
                self.insert(word, frequency)
    
    def search_exact(self, word: str) -> bool:
        """
        Check if an exact word exists in the Trie.
        
        Args:
            word: Word to search for
            
        Returns:
            True if word exists, False otherwise
            
        Time Complexity: O(m) where m is the length of the word
        """
        with self._lock:
            word = self._normalize_text(word.strip())
            current = self.root
            
            for char in word:
                if char not in current.children:
                    return False
                current = current.children[char]
            
            return current.is_end_word
    
    def search_prefix(self, prefix: str) -> List[Tuple[str, int]]:
        """
        Find all words that start with the given prefix.
        
        Args:
            prefix: The prefix to search for
            
        Returns:
            List of (word, frequency) tuples sorted by frequency (descending)
            
        Time Complexity: O(p + n) where p is prefix length and n is number of results
        """
        if not prefix.strip():
            return []
            
        with self._lock:
            prefix = self._normalize_text(prefix.strip())
            current = self.root
            
            # Navigate to the prefix node
            for char in prefix:
                if char not in current.children:
                    return []
                current = current.children[char]
            
            # Collect all words that start with this prefix
            suggestions = []
            self._collect_words(current, suggestions)
            
            # Sort by frequency (descending) and limit results
            suggestions.sort(key=lambda x: x[1], reverse=True)
            return suggestions[:self.max_suggestions]
    
    def _collect_words(self, node: TrieNode, results: List[Tuple[str, int]]) -> None:
        """
        Recursively collect all words from a given node.
        
        Args:
            node: Starting node for collection
            results: List to append results to
        """
        if node.is_end_word:
            results.append((node.word, node.frequency))
        
        for child in node.children.values():
            self._collect_words(child, results)
    
    def get_top_k_by_frequency(self, k: int = 10) -> List[Tuple[str, int]]:
        """
        Get the top k most frequent words in the Trie.
        
        Args:
            k: Number of top words to return
            
        Returns:
            List of (word, frequency) tuples sorted by frequency
        """
        with self._lock:
            all_words = []
            self._collect_words(self.root, all_words)
            
            # Use heap for efficient top-k selection
            if len(all_words) <= k:
                return sorted(all_words, key=lambda x: x[1], reverse=True)
            
            return heapq.nlargest(k, all_words, key=lambda x: x[1])
    
    def update_frequency(self, word: str, new_frequency: int) -> bool:
        """
        Update the frequency of an existing word.
        
        Args:
            word: Word to update
            new_frequency: New frequency value
            
        Returns:
            True if word was found and updated, False otherwise
        """
        with self._lock:
            word = self._normalize_text(word.strip())
            current = self.root
            
            for char in word:
                if char not in current.children:
                    return False
                current = current.children[char]
            
            if current.is_end_word:
                current.frequency = new_frequency
                return True
            return False
    
    def remove(self, word: str) -> bool:
        """
        Remove a word from the Trie.
        
        Args:
            word: Word to remove
            
        Returns:
            True if word was found and removed, False otherwise
            
        Note: This is a complex operation that may require tree restructuring
        """
        with self._lock:
            word = self._normalize_text(word.strip())
            found, _ = self._remove_recursive(self.root, word, 0)
            return found
    
    def _remove_recursive(self, node: TrieNode, word: str, index: int) -> Tuple[bool, bool]:
        """
        Recursively remove a word from the Trie.
        
        Args:
            node: Current node
            word: Word to remove
            index: Current character index
            
        Returns:
            Tuple of (word_found, should_delete_node)
        """
        if index == len(word):
            if not node.is_end_word:
                return False, False
            
            node.is_end_word = False
            node.word = None
            node.frequency = 0
            self._word_count -= 1
            
            # Return (word_found=True, should_delete=True if no children)
            return True, len(node.children) == 0
        
        char = word[index]
        if char not in node.children:
            return False, False
        
        word_found, should_delete_child = self._remove_recursive(
            node.children[char], word, index + 1
        )
        
        if should_delete_child:
            del node.children[char]
        
        # This node should be deleted if it's not an end word and has no children
        should_delete_this = not node.is_end_word and len(node.children) == 0
        
        return word_found, should_delete_this
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get Trie statistics for monitoring and debugging.
        
        Returns:
            Dictionary with various Trie metrics
        """
        with self._lock:
            def count_nodes(node: TrieNode) -> int:
                count = 1
                for child in node.children.values():
                    count += count_nodes(child)
                return count
            
            return {
                'word_count': self._word_count,
                'node_count': count_nodes(self.root),
                'max_suggestions': self.max_suggestions,
                'case_sensitive': self.case_sensitive
            }
    
    def clear(self) -> None:
        """Clear all data from the Trie."""
        with self._lock:
            self.root = TrieNode()
            self._word_count = 0


# TODO: Consider implementing the following optimizations:
# 1. DAFSA (Directed Acyclic Finite State Automaton) for better memory efficiency
# 2. Compressed Trie (Patricia Trie) for sparse datasets
# 3. Persistent storage support (pickle/JSON serialization)
# 4. Fuzzy matching with edit distance
# 5. Incremental learning from user interactions
# 6. Memory usage optimization with __slots__
# 7. Async/await support for I/O operations
