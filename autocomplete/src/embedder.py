"""
Sentence Embedder using Contrastive Learning for Semantic Autocomplete

This module provides a PyTorch-based sentence encoder that learns meaningful
representations of chat messages for semantic similarity search.

Features:
- Contrastive learning with positive/negative pairs
- Support for multiple backbone models (BERT, MiniLM, BGE, etc.)
- Batch processing for efficient training and inference
- GPU acceleration support
- Fine-tuning on chat-specific data

Author: Generated for Sup Chat Application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import math
import time
import random
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)
import logging
from dataclasses import dataclass
import json
import pickle
from pathlib import Path


@dataclass
class EmbedderConfig:
    """Configuration for the sentence embedder."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 128
    embedding_dim: int = 384
    temperature: float = 0.07
    margin: float = 0.5
    learning_rate: float = 2e-5
    batch_size: int = 32
    dropout: float = 0.1
    pooling_strategy: str = "mean"  # mean, cls, max
    normalize_embeddings: bool = True
    device: str = "auto"  # auto, cpu, cuda


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with chat message pairs.
    
    Expected data format:
    - Positive pairs: similar messages that should have similar embeddings
    - Negative pairs: dissimilar messages that should have different embeddings
    """
    
    def __init__(
        self, 
        positive_pairs: List[Tuple[str, str]], 
        negative_pairs: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        """
        Initialize the contrastive dataset.
        
        Args:
            positive_pairs: List of (text1, text2) tuples for similar messages
            negative_pairs: List of (text1, text2) tuples for dissimilar messages
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.pairs = []
        self.labels = []
        
        # Add positive pairs (label = 1)
        for text1, text2 in positive_pairs:
            self.pairs.append((text1, text2))
            self.labels.append(1)
            
        # Add negative pairs (label = 0)
        for text1, text2 in negative_pairs:
            self.pairs.append((text1, text2))
            self.labels.append(0)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]
        
        # Tokenize both texts
        encoding1 = self.tokenizer(
            text1,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            text2,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(),
            'attention_mask_1': encoding1['attention_mask'].squeeze(),
            'input_ids_2': encoding2['input_ids'].squeeze(),
            'attention_mask_2': encoding2['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SentenceEmbedder(nn.Module):
    """
    Sentence embedder using contrastive learning.
    
    This model learns to embed sentences into a semantic vector space where
    similar messages are close and dissimilar messages are far apart.
    """
    
    def __init__(self, config: EmbedderConfig):
        """
        Initialize the sentence embedder.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.config = config
        
        # Load backbone model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(config.model_name)
        
        # Get actual embedding dimension from backbone
        backbone_config = AutoConfig.from_pretrained(config.model_name)
        backbone_dim = backbone_config.hidden_size
        
        # Projection head to desired embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize projection head weights."""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _pool_embeddings(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool token embeddings to get sentence embeddings.
        
        Args:
            hidden_states: Token embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Sentence embeddings [batch_size, hidden_dim]
        """
        if self.config.pooling_strategy == "cls":
            # Use [CLS] token embedding
            return hidden_states[:, 0, :]
        
        elif self.config.pooling_strategy == "mean":
            # Mean pooling with attention mask
            masked_embeddings = hidden_states * attention_mask.unsqueeze(-1)
            sum_embeddings = masked_embeddings.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            return sum_embeddings / sum_mask
        
        elif self.config.pooling_strategy == "max":
            # Max pooling with attention mask
            masked_embeddings = hidden_states.masked_fill(
                attention_mask.unsqueeze(-1) == 0, float('-inf')
            )
            return masked_embeddings.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
    
    def encode(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input text into embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Normalized embeddings [batch_size, embedding_dim]
        """
        # Get backbone embeddings
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Pool to sentence embeddings
        pooled = self._pool_embeddings(outputs.last_hidden_state, attention_mask)
        
        # Project to target dimension
        embeddings = self.projection(pooled)
        
        # Normalize if requested
        if self.config.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(
        self, 
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            input_ids_1: First text token IDs
            attention_mask_1: First text attention mask
            input_ids_2: Second text token IDs
            attention_mask_2: Second text attention mask
            
        Returns:
            Tuple of (embeddings_1, embeddings_2)
        """
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)
        return emb1, emb2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training sentence embedders.
    
    Pulls similar pairs together and pushes dissimilar pairs apart.
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature scaling for similarity computation
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self, 
        embeddings_1: torch.Tensor, 
        embeddings_2: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings_1: First set of embeddings
            embeddings_2: Second set of embeddings
            labels: Binary labels (1 for similar, 0 for dissimilar)
            
        Returns:
            Contrastive loss value
        """
        # Compute cosine similarity
        similarity = F.cosine_similarity(embeddings_1, embeddings_2, dim=1)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        # Contrastive loss
        positive_loss = labels * (1 - similarity)
        negative_loss = (1 - labels) * torch.clamp(similarity - self.margin, min=0)
        
        loss = positive_loss + negative_loss
        return loss.mean()


class EmbedderTrainer:
    """
    Trainer for the sentence embedder using contrastive learning.
    """
    
    def __init__(
        self, 
        model: SentenceEmbedder, 
        config: EmbedderConfig,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SentenceEmbedder model
            config: Configuration object
            device: Device to use for training
        """
        self.model = model
        self.config = config
        
        # Set device
        if device is None:
            if config.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(config.device)
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        self.criterion = ContrastiveLoss(
            temperature=config.temperature,
            margin=config.margin
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            emb1, emb2 = self.model(
                batch['input_ids_1'],
                batch['attention_mask_1'],
                batch['input_ids_2'],
                batch['attention_mask_2']
            )
            
            # Compute loss
            loss = self.criterion(emb1, emb2, batch['label'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                emb1, emb2 = self.model(
                    batch['input_ids_1'],
                    batch['attention_mask_1'],
                    batch['input_ids_2'],
                    batch['attention_mask_2']
                )
                
                # Compute loss
                loss = self.criterion(emb1, emb2, batch['label'])
                total_loss += loss.item()
                
                # Compute accuracy based on similarity threshold
                similarities = F.cosine_similarity(emb1, emb2, dim=1)
                predictions = (similarities > 0.5).float()
                correct_predictions += (predictions == batch['label']).sum().item()
                total_predictions += len(batch['label'])
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_model(self, path: Union[str, Path]):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {path}")


def generate_training_pairs_from_chat_data(
    messages: List[str],
    strategy: str = "similarity"
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Generate positive and negative pairs from chat messages.
    
    Args:
        messages: List of chat messages
        strategy: Strategy for pair generation ("similarity", "random", "temporal")
        
    Returns:
        Tuple of (positive_pairs, negative_pairs)
    """
    positive_pairs = []
    negative_pairs = []
    
    if strategy == "similarity":
        # Simple similarity based on shared words
        for i, msg1 in enumerate(messages):
            for j, msg2 in enumerate(messages[i+1:], i+1):
                words1 = set(msg1.lower().split())
                words2 = set(msg2.lower().split())
                overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                
                if overlap > 0.3:  # Similar messages
                    positive_pairs.append((msg1, msg2))
                elif overlap < 0.1:  # Dissimilar messages
                    negative_pairs.append((msg1, msg2))
    
    elif strategy == "random":
        # Random positive and negative pairs
        import random
        n_pairs = min(len(messages) // 2, 1000)
        
        for _ in range(n_pairs):
            idx1, idx2 = random.sample(range(len(messages)), 2)
            if random.random() > 0.5:
                positive_pairs.append((messages[idx1], messages[idx2]))
            else:
                negative_pairs.append((messages[idx1], messages[idx2]))
    
    # Ensure balanced dataset
    min_pairs = min(len(positive_pairs), len(negative_pairs))
    positive_pairs = positive_pairs[:min_pairs]
    negative_pairs = negative_pairs[:min_pairs]
    
    return positive_pairs, negative_pairs


# TODO: Implement the following enhancements:
# 1. Multi-task learning with classification heads
# 2. Hard negative mining for better contrastive learning
# 3. Curriculum learning with increasing difficulty
# 4. Knowledge distillation from larger models
# 5. Cross-lingual embeddings for multilingual chat
# 6. Temporal embeddings for conversation context
# 7. User-specific embeddings for personalization
# 8. Online learning for continuous adaptation
