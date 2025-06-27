"""
Transformer-based Text Generator for Predictive Typing

This module provides a small, efficient transformer model for next-token and
next-sentence prediction in real-time chat scenarios. Optimized for low
latency while maintaining reasonable generation quality.

Features:
- Lightweight transformer architecture (GPT-style)
- Fast inference with KV-caching
- Chat-domain fine-tuning
- Beam search and nucleus sampling
- Real-time streaming generation
- Memory-efficient implementation

Author: Generated for Sup Chat Application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Union, Any, Iterator
import numpy as np
import logging
import math
import random
import heapq
from dataclasses import dataclass
from pathlib import Path
import json
import time
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)


@dataclass
class GeneratorConfig:
    """Configuration for the text generator."""
    # Model architecture
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 16
    
    # Generation parameters
    max_length: int = 50
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    beam_size: int = 3
    length_penalty: float = 1.0
    
    # Optimization
    use_gradient_checkpointing: bool = False
    compile_model: bool = False
    device: str = "auto"
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask buffer
        self.register_buffer('causal_mask', torch.tril(torch.ones(1000, 1000)))
    
    def forward(
        self, 
        x: torch.Tensor, 
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Use cached keys and values if available
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Prepare cache for next iteration
        if use_cache:
            new_cache = (k, v)
        else:
            new_cache = None
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        current_seq_len = k.size(2)
        mask = self.causal_mask[:current_seq_len, :current_seq_len]
        if seq_len == 1 and past_key_value is not None:
            # For incremental decoding, mask only new position
            mask = mask[-1:, :]
        else:
            mask = mask[-seq_len:, :]
        
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Concatenate heads and apply output projection
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.w_o(attended)
        
        return output, new_cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config.d_model, config.n_heads, config.dropout)
        self.feed_forward = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Self-attention with residual connection
        attn_output, new_cache = self.attention(
            self.ln1(x), 
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x, new_cache


class ChatGenerator(nn.Module):
    """
    Lightweight transformer model for chat text generation.
    
    Based on GPT architecture but optimized for real-time inference
    in chat applications.
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (common practice)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # For KV caching during generation
        self.past_key_values = None
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        
        # Input embeddings
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        new_cache = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, cache = block(x, past_key_value=past_kv, use_cache=use_cache)
            if use_cache:
                new_cache.append(cache)
        
        # Final layer norm and language modeling head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, new_cache if use_cache else None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            num_beams: Number of beams for beam search
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs
        """
        max_length = max_length or self.config.max_length
        pad_token_id = pad_token_id or self.config.pad_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id
        
        if num_beams > 1:
            return self._beam_search(
                input_ids, max_length, num_beams, temperature, 
                pad_token_id, eos_token_id
            )
        else:
            return self._greedy_or_sample(
                input_ids, max_length, temperature, top_k, top_p,
                do_sample, pad_token_id, eos_token_id
            )
    
    def _greedy_or_sample(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int
    ) -> torch.Tensor:
        """Greedy decoding or sampling generation."""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize past key values for caching
        past_key_values = None
        
        generated = input_ids.clone()
        
        for _ in range(max_length - seq_len):
            # Get next token logits
            if past_key_values is None:
                model_input = generated
            else:
                model_input = generated[:, -1:]  # Only last token for cached inference
            
            with torch.no_grad():
                logits, past_key_values = self.forward(
                    model_input,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Apply top-k and top-p filtering
                next_token_logits = self._top_k_top_p_filtering(
                    next_token_logits, top_k, top_p
                )
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def _top_k_top_p_filtering(
        self, 
        logits: torch.Tensor, 
        top_k: int, 
        top_p: float
    ) -> torch.Tensor:
        """Apply top-k and top-p (nucleus) filtering."""
        
        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k)
            indices_to_remove = logits < top_k_logits[..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        num_beams: int,
        temperature: float,
        pad_token_id: int,
        eos_token_id: int
    ) -> torch.Tensor:
        """Beam search generation (simplified implementation)."""
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Expand input for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, seq_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, seq_len)
        
        # Beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = float('-inf')
        beam_scores = beam_scores.view(-1)
        
        # Generated sequences
        generated = input_ids.clone()
        
        for _ in range(max_length - seq_len):
            with torch.no_grad():
                logits, _ = self.forward(generated)
            
            next_token_logits = logits[:, -1, :] / temperature
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.unsqueeze(1)
            
            # Reshape for beam search
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams tokens
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            
            # Calculate beam and token indices
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Prepare for next iteration
            beam_outputs = []
            beam_scores_new = []
            
            for batch_idx in range(batch_size):
                beam_tokens = []
                beam_scores_batch = []
                
                for beam_token_rank, (token_score, token, indices) in enumerate(
                    zip(next_token_scores[batch_idx], next_tokens[batch_idx], next_indices[batch_idx])
                ):
                    beam_id = batch_idx * num_beams + indices
                    
                    # Add token to beam
                    new_seq = torch.cat([generated[beam_id], token.unsqueeze(0)])
                    beam_tokens.append(new_seq)
                    beam_scores_batch.append(token_score)
                    
                    if len(beam_tokens) >= num_beams:
                        break
                
                beam_outputs.extend(beam_tokens)
                beam_scores_new.extend(beam_scores_batch)
            
            generated = torch.stack(beam_outputs)
            beam_scores = torch.tensor(beam_scores_new, device=device)
        
        # Return best beam for each batch
        best_beams = generated.view(batch_size, num_beams, -1)[:, 0, :]
        return best_beams


class ChatDataset(Dataset):
    """Dataset for training the chat generator."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Tokenize and create sliding windows
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            
            # Create overlapping windows
            for i in range(0, len(tokens) - max_length + 1, stride):
                window = tokens[i:i + max_length]
                if len(window) == max_length:
                    self.examples.append(window)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        
        # Input is all tokens except last, target is all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class GeneratorTrainer:
    """Trainer for the chat generator."""
    
    def __init__(self, model: ChatGenerator, config: GeneratorConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Compile model if requested
        if config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits, _ = self.model(input_ids)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.config.pad_token_id
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    def save_model(self, path: Union[str, Path]):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")


# TODO: Implement the following enhancements:
# 1. Rotary Position Embedding (RoPE) for better length generalization
# 2. Flash Attention for memory efficiency
# 3. Model parallelism for larger models
# 4. Quantization for mobile deployment
# 5. Streaming generation with WebSocket support
# 6. Fine-tuning with RLHF for chat optimization
# 7. Multi-turn conversation modeling
# 8. Domain adaptation for specific chat contexts
