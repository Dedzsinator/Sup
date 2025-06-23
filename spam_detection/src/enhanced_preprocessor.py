"""
Enhanced Data Preprocessing Pipeline for Spam Detection

This module provides comprehensive preprocessing capabilities for various data sources:
- Individual email files (spam/ham directories)
- SMS data (SMSSpamCollection)
- Text data files (train.txt, test.txt)

Features include:
- Email header analysis and extraction
- Advanced text preprocessing with NLP techniques
- Feature engineering (TF-IDF, n-grams, linguistic features)
- Semi-supervised learning data preparation
- Cross-validation dataset splitting
"""

import os
import re
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
import email
from email.parser import Parser
from email.policy import default
import warnings

# Progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create a dummy tqdm class
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable or []
            self.desc = desc
            self.total = total
        
        def __iter__(self):
            return iter(self.iterable)
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass
        
        def set_description(self, desc):
            pass
        
        def close(self):
            pass

# NLP and ML imports
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Some features will be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. ML features will be limited.")

logger = logging.getLogger(__name__)

class EnhancedDataPreprocessor:
    """
    Advanced data preprocessing pipeline for spam detection
    """
    
    def __init__(self, 
                 data_dir: str = "/home/deginandor/Documents/Programming/Sup/spam_detection/data",
                 stopwords_path: Optional[str] = None,
                 enable_nltk: bool = True,
                 enable_advanced_features: bool = True):
        
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_nltk = enable_nltk and NLTK_AVAILABLE
        self.enable_advanced_features = enable_advanced_features and SKLEARN_AVAILABLE
        
        # Initialize NLP tools
        if self.enable_nltk:
            self._init_nltk_tools()
        
        # Load custom stopwords
        self.stopwords = self._load_stopwords(stopwords_path)
        
        # Initialize feature extractors
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.scaler = None
        
        logger.info(f"Initialized EnhancedDataPreprocessor with data_dir={data_dir}")
        logger.info(f"NLTK enabled: {self.enable_nltk}")
        logger.info(f"Advanced features enabled: {self.enable_advanced_features}")
    
    def _init_nltk_tools(self):
        """Initialize NLTK tools and download required data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        try:
            nltk.data.find('averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        try:
            nltk.data.find('maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker', quiet=True)
        
        try:
            nltk.data.find('words')
        except LookupError:
            nltk.download('words', quiet=True)
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def _load_stopwords(self, stopwords_path: Optional[str] = None) -> set:
        """Load stopwords from file or NLTK"""
        stopwords_set = set()
        
        # Load from NLTK
        if self.enable_nltk:
            stopwords_set.update(stopwords.words('english'))
        
        # Load custom stopwords
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                custom_stopwords = f.read().splitlines()
                stopwords_set.update(custom_stopwords)
        
        # Load from data directory
        for stopwords_file in ['stopwords.txt', 'stopwords2.txt']:
            path = self.raw_dir / stopwords_file
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    custom_stopwords = f.read().splitlines()
                    stopwords_set.update(custom_stopwords)
        
        logger.info(f"Loaded {len(stopwords_set)} stopwords")
        return stopwords_set
    
    def load_email_data(self) -> Tuple[List[str], List[int], List[Dict]]:
        """
        Load all email data from spam and ham directories
        
        Returns:
            Tuple of (messages, labels, metadata)
        """
        messages = []
        labels = []
        metadata = []
        
        # Load spam emails
        spam_dir = self.raw_dir / "spam"
        if spam_dir.exists():
            spam_files = list(spam_dir.glob("*.txt"))
            logger.info(f"Loading {len(spam_files)} spam email files...")
            
            for file_path in tqdm(spam_files, desc="Loading spam emails", unit="files"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Parse email if possible
                    email_data = self._parse_email(content)
                    messages.append(email_data['body'])
                    labels.append(1)  # spam
                    
                    # Extract metadata from filename
                    file_metadata = self._parse_filename(file_path.name)
                    file_metadata.update(email_data['headers'])
                    file_metadata['source'] = 'email'
                    file_metadata['file_path'] = str(file_path)
                    metadata.append(file_metadata)
                    
                except Exception as e:
                    logger.warning(f"Error processing spam file {file_path}: {e}")
                    continue
        
        # Load ham emails
        ham_dir = self.raw_dir / "ham"
        if ham_dir.exists():
            ham_files = list(ham_dir.glob("*.txt"))
            logger.info(f"Loading {len(ham_files)} ham email files...")
            
            for file_path in tqdm(ham_files, desc="Loading ham emails", unit="files"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Parse email if possible
                    email_data = self._parse_email(content)
                    messages.append(email_data['body'])
                    labels.append(0)  # ham
                    
                    # Extract metadata from filename
                    file_metadata = self._parse_filename(file_path.name)
                    file_metadata.update(email_data['headers'])
                    file_metadata['source'] = 'email'
                    file_metadata['file_path'] = str(file_path)
                    metadata.append(file_metadata)
                    
                except Exception as e:
                    logger.warning(f"Error processing ham file {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(messages)} email messages ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")
        return messages, labels, metadata
    
    def load_sms_data(self) -> Tuple[List[str], List[int], List[Dict]]:
        """
        Load SMS data from SMSSpamCollection file
        
        Returns:
            Tuple of (messages, labels, metadata)
        """
        messages = []
        labels = []
        metadata = []
        
        sms_file = self.raw_dir / "SMSSpamCollection"
        if not sms_file.exists():
            logger.warning("SMSSpamCollection file not found")
            return messages, labels, metadata
        
        with open(sms_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        logger.info(f"Loading {len(lines)} SMS messages...")
        
        for line_num, line in enumerate(tqdm(lines, desc="Loading SMS messages", unit="messages"), 1):
            try:
                parts = line.strip().split('\t', 1)
                if len(parts) != 2:
                    continue
                
                label_str, message = parts
                label = 1 if label_str.lower() == 'spam' else 0
                
                messages.append(message)
                labels.append(label)
                metadata.append({
                    'source': 'sms',
                    'line_number': line_num,
                    'original_label': label_str
                })
                
            except Exception as e:
                logger.warning(f"Error processing SMS line {line_num}: {e}")
                continue
        
        logger.info(f"Loaded {len(messages)} SMS messages ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")
        return messages, labels, metadata
    
    def load_text_files(self) -> Tuple[List[str], List[int], List[Dict]]:
        """
        Load data from train.txt and test.txt files
        
        Returns:
            Tuple of (messages, labels, metadata)
        """
        messages = []
        labels = []
        metadata = []
        
        for filename in ['train.txt', 'test.txt']:
            file_path = self.raw_dir / filename
            if not file_path.exists():
                continue
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Try to parse as label\tmessage format
                        if '\t' in line:
                            parts = line.split('\t', 1)
                            if len(parts) == 2:
                                label_str, message = parts
                                label = 1 if label_str.lower() in ['spam', '1'] else 0
                            else:
                                # Assume it's just a message, label as unknown
                                message = line
                                label = -1  # Unknown label for semi-supervised learning
                        else:
                            message = line
                            label = -1  # Unknown label for semi-supervised learning
                        
                        messages.append(message)
                        labels.append(label)
                        metadata.append({
                            'source': 'text_file',
                            'filename': filename,
                            'line_number': line_num
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num} in {filename}: {e}")
                        continue
        
        logger.info(f"Loaded {len(messages)} messages from text files")
        return messages, labels, metadata
    
    def _parse_email(self, content: str) -> Dict:
        """Parse email content and extract headers and body"""
        try:
            # Try parsing as proper email
            if content.startswith('Subject:') or 'From:' in content[:200]:
                msg = email.message_from_string(content, policy=default)
                
                headers = {}
                for key, value in msg.items():
                    headers[key.lower()] = value
                
                # Extract body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                else:
                    body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                
                return {
                    'headers': headers,
                    'body': body.strip()
                }
            else:
                # Content doesn't look like proper email, treat as plain text
                lines = content.split('\n')
                headers = {}
                body_lines = []
                
                # Look for subject line
                for i, line in enumerate(lines):
                    if line.lower().startswith('subject:'):
                        headers['subject'] = line[8:].strip()
                        body_lines = lines[i+1:]
                        break
                else:
                    body_lines = lines
                
                return {
                    'headers': headers,
                    'body': '\n'.join(body_lines).strip()
                }
                
        except Exception as e:
            logger.debug(f"Email parsing failed: {e}")
            return {
                'headers': {},
                'body': content.strip()
            }
    
    def _parse_filename(self, filename: str) -> Dict:
        """Parse metadata from filename like '0001.2000-06-06.lokay.ham.txt'"""
        metadata = {'filename': filename}
        
        # Try to extract date, user, and type from filename
        parts = filename.split('.')
        if len(parts) >= 4:
            try:
                metadata['message_id'] = parts[0]
                metadata['date'] = parts[1]
                metadata['user'] = parts[2]
                metadata['type'] = parts[3]
            except:
                pass
        
        return metadata
    
    def preprocess_text(self, text: str, 
                       lowercase: bool = True,
                       remove_punctuation: bool = True,
                       remove_numbers: bool = False,
                       remove_stopwords: bool = True,
                       stem: bool = False,
                       lemmatize: bool = True) -> str:
        """
        Comprehensive text preprocessing
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', ' PHONE ', text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenization and further processing
        if self.enable_nltk:
            tokens = word_tokenize(text)
            
            # Remove punctuation
            if remove_punctuation:
                tokens = [token for token in tokens if token.isalnum()]
            
            # Remove numbers
            if remove_numbers:
                tokens = [token for token in tokens if not token.isdigit()]
            
            # Remove stopwords
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stopwords]
            
            # Stemming
            if stem and hasattr(self, 'stemmer'):
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            # Lemmatization
            if lemmatize and hasattr(self, 'lemmatizer'):
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            return ' '.join(tokens)
        else:
            # Basic preprocessing without NLTK
            if remove_punctuation:
                text = re.sub(r'[^\w\s]', ' ', text)
            
            if remove_numbers:
                text = re.sub(r'\d+', ' ', text)
            
            if remove_stopwords:
                words = text.split()
                words = [word for word in words if word not in self.stopwords]
                text = ' '.join(words)
            
            return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from text
        """
        features = {}
        
        if not text:
            return {f'ling_{k}': 0.0 for k in ['length', 'words', 'sentences', 'avg_word_len', 
                                              'caps_ratio', 'punct_ratio', 'digit_ratio', 'exclamation_ratio']}
        
        # Basic length features
        features['ling_length'] = len(text)
        words = text.split()
        features['ling_words'] = len(words)
        
        # Sentence count
        if self.enable_nltk:
            sentences = sent_tokenize(text)
            features['ling_sentences'] = len(sentences)
        else:
            features['ling_sentences'] = len(re.split(r'[.!?]+', text))
        
        # Average word length
        if words:
            features['ling_avg_word_len'] = sum(len(word) for word in words) / len(words)
        else:
            features['ling_avg_word_len'] = 0.0
        
        # Character ratios
        features['ling_caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0
        features['ling_punct_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0.0
        features['ling_digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0.0
        features['ling_exclamation_ratio'] = text.count('!') / len(text) if text else 0.0
        
        return features
    
    def extract_spam_features(self, text: str) -> Dict[str, float]:
        """
        Extract spam-specific features
        """
        features = {}
        
        if not text:
            return {f'spam_{k}': 0.0 for k in ['urgency', 'money', 'free', 'caps', 'phone', 'url', 'winner']}
        
        text_lower = text.lower()
        
        # Urgency indicators
        urgency_words = ['urgent', 'act now', 'limited time', 'expires', 'hurry', 'rush', 'immediate']
        features['spam_urgency'] = sum(text_lower.count(word) for word in urgency_words)
        
        # Money-related terms
        money_patterns = [r'\$\d+', r'money', r'cash', r'prize', r'win', r'earn', r'income']
        features['spam_money'] = sum(len(re.findall(pattern, text_lower)) for pattern in money_patterns)
        
        # Free offers
        free_words = ['free', 'no cost', 'complimentary', 'gratis']
        features['spam_free'] = sum(text_lower.count(word) for word in free_words)
        
        # Excessive capitalization
        features['spam_caps'] = 1 if sum(1 for c in text if c.isupper()) > len(text) * 0.3 else 0
        
        # Contact information
        features['spam_phone'] = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        features['spam_url'] = len(re.findall(r'http[s]?://', text_lower))
        
        # Winner/lottery terms
        winner_words = ['winner', 'congratulations', 'selected', 'lottery', 'jackpot']
        features['spam_winner'] = sum(text_lower.count(word) for word in winner_words)
        
        return features
    
    def create_feature_matrix(self, messages: List[str], 
                             max_features: int = 10000,
                             ngram_range: Tuple[int, int] = (1, 2),
                             use_tfidf: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix from messages with TF-IDF and linguistic features
        """
        if not self.enable_advanced_features:
            # Fallback to basic features
            return self._create_basic_features(messages)
        
        # Preprocess messages
        logger.info("Preprocessing messages...")
        processed_messages = []
        for msg in tqdm(messages, desc="Preprocessing text", unit="messages"):
            processed_messages.append(self.preprocess_text(msg))
        
        # TF-IDF features
        logger.info("Creating TF-IDF features...")
        if use_tfidf:
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words='english',
                    lowercase=True,
                    strip_accents='unicode'
                )
                tfidf_features = self.tfidf_vectorizer.fit_transform(processed_messages)
            else:
                tfidf_features = self.tfidf_vectorizer.transform(processed_messages)
        else:
            if self.count_vectorizer is None:
                self.count_vectorizer = CountVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words='english',
                    lowercase=True,
                    strip_accents='unicode'
                )
                tfidf_features = self.count_vectorizer.fit_transform(processed_messages)
            else:
                tfidf_features = self.count_vectorizer.transform(processed_messages)
        
        # Extract linguistic and spam features
        logger.info("Extracting linguistic and spam features...")
        linguistic_features = []
        spam_features = []
        
        for msg in tqdm(messages, desc="Extracting features", unit="messages"):
            ling_feat = self.extract_linguistic_features(msg)
            spam_feat = self.extract_spam_features(msg)
            
            linguistic_features.append(list(ling_feat.values()))
            spam_features.append(list(spam_feat.values()))
        
        # Combine features
        linguistic_features = np.array(linguistic_features)
        spam_features = np.array(spam_features)
        
        # Scale linguistic and spam features
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(
                np.hstack([linguistic_features, spam_features])
            )
        else:
            scaled_features = self.scaler.transform(
                np.hstack([linguistic_features, spam_features])
            )
        
        # Combine TF-IDF with other features
        if hasattr(tfidf_features, 'toarray'):
            tfidf_features = tfidf_features.toarray()
        
        combined_features = np.hstack([tfidf_features, scaled_features])
        
        # Feature names for interpretability
        feature_names = []
        if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            feature_names.extend(self.tfidf_vectorizer.get_feature_names_out())
        elif hasattr(self.count_vectorizer, 'get_feature_names_out'):
            feature_names.extend(self.count_vectorizer.get_feature_names_out())
        
        # Add linguistic feature names
        ling_feat_sample = self.extract_linguistic_features("sample")
        spam_feat_sample = self.extract_spam_features("sample")
        feature_names.extend(list(ling_feat_sample.keys()))
        feature_names.extend(list(spam_feat_sample.keys()))
        
        return combined_features, np.array(feature_names)
    
    def _create_basic_features(self, messages: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create basic features when sklearn is not available"""
        features = []
        feature_names = []
        
        for msg in messages:
            msg_features = []
            
            # Basic text features
            msg_features.extend([
                len(msg),  # length
                len(msg.split()),  # word count
                msg.count('!'),  # exclamation marks
                msg.count('$'),  # dollar signs
                sum(1 for c in msg if c.isupper()),  # uppercase letters
            ])
            
            # Add linguistic and spam features
            ling_feat = self.extract_linguistic_features(msg)
            spam_feat = self.extract_spam_features(msg)
            
            msg_features.extend(list(ling_feat.values()))
            msg_features.extend(list(spam_feat.values()))
            
            features.append(msg_features)
        
        # Feature names
        feature_names = ['basic_length', 'basic_words', 'basic_exclamation', 'basic_dollar', 'basic_caps']
        feature_names.extend(list(self.extract_linguistic_features("sample").keys()))
        feature_names.extend(list(self.extract_spam_features("sample").keys()))
        
        return np.array(features), np.array(feature_names)
    
    def load_all_data(self) -> Tuple[List[str], List[int], List[Dict]]:
        """
        Load all available data from all sources
        """
        all_messages = []
        all_labels = []
        all_metadata = []
        
        # Load email data
        email_messages, email_labels, email_metadata = self.load_email_data()
        all_messages.extend(email_messages)
        all_labels.extend(email_labels)
        all_metadata.extend(email_metadata)
        
        # Load SMS data
        sms_messages, sms_labels, sms_metadata = self.load_sms_data()
        all_messages.extend(sms_messages)
        all_labels.extend(sms_labels)
        all_metadata.extend(sms_metadata)
        
        # Load text files
        text_messages, text_labels, text_metadata = self.load_text_files()
        all_messages.extend(text_messages)
        all_labels.extend(text_labels)
        all_metadata.extend(text_metadata)
        
        logger.info(f"Total loaded: {len(all_messages)} messages")
        logger.info(f"Label distribution: {np.bincount(np.array(all_labels)[np.array(all_labels) >= 0])}")
        
        return all_messages, all_labels, all_metadata
    
    def prepare_cross_validation_splits(self, messages: List[str], labels: List[int], 
                                      n_splits: int = 5, random_state: int = 42) -> List[Tuple]:
        """
        Prepare cross-validation splits with stratification
        """
        if not self.enable_advanced_features:
            # Simple random splits
            indices = np.arange(len(messages))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            splits = []
            fold_size = len(indices) // n_splits
            
            for i in range(n_splits):
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(indices)
                
                test_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                splits.append((train_indices, test_indices))
            
            return splits
        
        # Use sklearn's StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(skf.split(messages, labels))
    
    def save_processed_data(self, messages: List[str], labels: List[int], 
                           metadata: List[Dict], filename: str = "processed_data.jsonl"):
        """
        Save processed data to file
        """
        output_path = self.processed_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for msg, label, meta in zip(messages, labels, metadata):
                data = {
                    'message': msg,
                    'label': label,
                    'metadata': meta,
                    'processed_at': datetime.now().isoformat()
                }
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Saved {len(messages)} processed messages to {output_path}")
    
    def load_processed_data(self, filename: str = "processed_data.jsonl") -> Tuple[List[str], List[int], List[Dict]]:
        """
        Load processed data from file
        """
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Processed data file {file_path} not found")
            return [], [], []
        
        messages = []
        labels = []
        metadata = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    messages.append(data['message'])
                    labels.append(data['label'])
                    metadata.append(data['metadata'])
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(messages)} processed messages from {file_path}")
        return messages, labels, metadata
