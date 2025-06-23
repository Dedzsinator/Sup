"""
Utilities for spam detection system

This module provides utility functions for data processing, feature engineering,
and performance optimization.
"""

import re
import string
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import unicodedata
import hashlib
import time
from functools import wraps
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Advanced text preprocessing for spam detection"""
    
    def __init__(self):
        # Common spam patterns
        self.spam_patterns = {
            'urgency': [
                r'\b(urgent|asap|immediate|now|hurry|quick|fast)\b',
                r'\b(limited time|expires|deadline|act now)\b'
            ],
            'money': [
                r'\$\d+',
                r'\b(free|money|cash|prize|win|won|winner)\b',
                r'\b(\d+%\s*(off|discount|savings))\b'
            ],
            'suspicious_chars': [
                r'[!]{2,}',  # Multiple exclamation marks
                r'[?]{2,}',  # Multiple question marks
                r'[A-Z]{5,}',  # Long sequences of uppercase
                r'[\d]{10,}'  # Long sequences of digits
            ],
            'contact_info': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'  # URLs
            ]
        }
        
        # Initialize preprocessing components
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.compiled_patterns = {}
        for category, patterns in self.spam_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing"""
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_spam_features(self, text: str) -> Dict[str, float]:
        """Extract spam-specific features from text"""
        features = {}
        text_lower = text.lower()
        
        # Pattern-based features
        for category, patterns in self.compiled_patterns.items():
            count = 0
            for pattern in patterns:
                count += len(pattern.findall(text))
            features[f'{category}_count'] = count
            features[f'{category}_ratio'] = count / len(text.split()) if text.split() else 0
        
        # Character-based features
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        features['punct_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        
        # Length features
        words = text.split()
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Repetition features
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        if word_counts:
            max_repetition = max(word_counts.values())
            features['max_word_repetition'] = max_repetition
            features['repetition_ratio'] = max_repetition / len(words)
        else:
            features['max_word_repetition'] = 0
            features['repetition_ratio'] = 0
        
        return features
    
    def clean_text(self, text: str) -> str:
        """Clean text for tokenization"""
        # Normalize first
        text = self.normalize_text(text)
        
        # Replace URLs with placeholder
        text = re.sub(r'http[s]?://\S+', '[URL]', text)
        
        # Replace emails with placeholder
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Replace phone numbers with placeholder
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Replace excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Replace numbers with placeholder (but keep small numbers)
        text = re.sub(r'\b\d{5,}\b', '[NUMBER]', text)
        
        return text

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def timing_decorator(func):
        """Decorator to measure function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} took {(end_time - start_time) * 1000:.2f}ms")
            return result
        return wrapper
    
    @staticmethod
    def memoize(maxsize: int = 128):
        """Simple memoization decorator with LRU cache"""
        def decorator(func):
            cache = {}
            cache_order = []
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = hashlib.md5(str((args, kwargs)).encode()).hexdigest()
                
                if key in cache:
                    # Move to end (most recently used)
                    cache_order.remove(key)
                    cache_order.append(key)
                    return cache[key]
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Add to cache
                cache[key] = result
                cache_order.append(key)
                
                # Remove oldest if cache is full
                if len(cache) > maxsize:
                    oldest_key = cache_order.pop(0)
                    del cache[oldest_key]
                
                return result
            
            return wrapper
        return decorator
    
    @staticmethod
    def batch_process(items: List, batch_size: int = 100):
        """Process items in batches for memory efficiency"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

class FeatureEngineering:
    """Advanced feature engineering for spam detection"""
    
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text"""
        words = text.split()
        if len(words) < n:
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = '_'.join(words[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def extract_character_ngrams(self, text: str, n: int = 3) -> List[str]:
        """Extract character-level n-grams"""
        if len(text) < n:
            return []
        
        char_ngrams = []
        for i in range(len(text) - n + 1):
            char_ngram = text[i:i + n]
            char_ngrams.append(char_ngram)
        
        return char_ngrams
    
    def extract_syntactic_features(self, text: str) -> Dict[str, float]:
        """Extract syntactic and structural features"""
        features = {}
        
        # Sentence-level features
        sentences = re.split(r'[.!?]+', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        
        # Part-of-speech patterns (simplified)
        words = text.split()
        
        # Count different types of words (basic heuristics)
        features['question_words'] = sum(1 for w in words if w.lower() in ['what', 'where', 'when', 'why', 'how', 'who'])
        features['action_words'] = sum(1 for w in words if w.lower() in ['click', 'call', 'buy', 'order', 'download', 'visit'])
        features['emotion_words'] = sum(1 for w in words if w.lower() in ['amazing', 'incredible', 'fantastic', 'urgent', 'limited'])
        
        # Punctuation patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        
        return features
    
    def extract_user_context_features(self, user_id: str, message_history: List[str]) -> Dict[str, float]:
        """Extract user context features from message history"""
        features = {}
        
        if not message_history:
            return {'user_message_count': 0, 'user_avg_length': 0, 'user_vocab_diversity': 0}
        
        # User behavior patterns
        features['user_message_count'] = len(message_history)
        features['user_avg_length'] = np.mean([len(msg) for msg in message_history])
        
        # Vocabulary diversity
        all_words = []
        for msg in message_history:
            all_words.extend(msg.split())
        
        unique_words = set(all_words)
        features['user_vocab_diversity'] = len(unique_words) / len(all_words) if all_words else 0
        
        # Temporal patterns (if timestamps available)
        # This would require timestamp data
        
        return features

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_message(message: str) -> bool:
        """Validate if message is suitable for processing"""
        if not message or not message.strip():
            return False
        
        # Check length limits
        if len(message) > 10000:  # Too long
            return False
        
        if len(message) < 3:  # Too short
            return False
        
        # Check for only special characters
        if re.match(r'^[^\w\s]+$', message):
            return False
        
        return True
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """Validate user ID format"""
        if not user_id or not user_id.strip():
            return False
        
        # Basic format validation
        if len(user_id) > 100:
            return False
        
        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return False
        
        return True
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize input text"""
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000]
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

class MetricsCollector:
    """Collect and manage performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
    
    def record_timing(self, operation: str, duration_ms: float):
        """Record timing metric"""
        self.metrics[f'{operation}_timing'].append(duration_ms)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment counter metric"""
        self.counters[counter_name] += value
    
    def get_avg_timing(self, operation: str) -> float:
        """Get average timing for operation"""
        timings = self.metrics.get(f'{operation}_timing', [])
        return np.mean(timings) if timings else 0
    
    def get_percentile_timing(self, operation: str, percentile: float = 95) -> float:
        """Get percentile timing for operation"""
        timings = self.metrics.get(f'{operation}_timing', [])
        return np.percentile(timings, percentile) if timings else 0
    
    def get_counter(self, counter_name: str) -> int:
        """Get counter value"""
        return self.counters.get(counter_name, 0)
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics"""
        result = {}
        
        # Add timing metrics
        for key, values in self.metrics.items():
            if values:
                result[f'{key}_avg'] = np.mean(values)
                result[f'{key}_p95'] = np.percentile(values, 95)
                result[f'{key}_count'] = len(values)
        
        # Add counters
        result.update(self.counters)
        
        return result
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()

# Global metrics collector instance
metrics_collector = MetricsCollector()

def get_metrics() -> Dict:
    """Get current metrics"""
    return metrics_collector.get_all_metrics()

def record_timing(operation: str, duration_ms: float):
    """Record timing metric"""
    metrics_collector.record_timing(operation, duration_ms)

def increment_counter(counter_name: str, value: int = 1):
    """Increment counter"""
    metrics_collector.increment_counter(counter_name, value)
