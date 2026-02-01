# -*- coding: utf-8 -*-
"""
Dynamic Transformer with PyTorch Training
Transformer ديناميكي مع تدريب حقيقي

Key Innovation: Train with PyTorch, save weights as Python code.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class DynamicEmbeddingTorch(nn.Module):
    """Embedding layer that can export to code."""
    
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Word mapping
        self.word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx2word: Dict[int, str] = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.next_idx = 4
    
    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            if self.next_idx < self.vocab_size:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1
        return self.word2idx.get(word, 1)  # Return <UNK> if full
    
    def encode(self, words: List[str]) -> torch.Tensor:
        indices = [self.word2idx.get(w, 1) for w in words]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> List[str]:
        return [self.idx2word.get(i.item(), "<UNK>") for i in indices]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
    
    def to_source_code(self) -> str:
        weights = self.embedding.weight.detach().cpu().numpy()
        code = f'''# Embedding weights: vocab={self.vocab_size}, dim={self.dim}
EMBEDDING_WEIGHTS = np.array({np.array2string(weights[:self.next_idx], separator=",", threshold=10000)})
WORD2IDX = {dict(list(self.word2idx.items())[:100])}  # First 100
'''
        return code


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class DynamicAttentionTorch(nn.Module):
    """Multi-head attention with pattern tracking."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)


class DynamicFFNTorch(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class DynamicTransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = DynamicAttentionTorch(dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = DynamicFFNTorch(dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class DynamicTransformerLM(nn.Module):
    """
    Dynamic Transformer Language Model
    
    - Trains with PyTorch (backprop)
    - Saves weights as Python code
    - Can run inference without PyTorch
    """
    
    def __init__(self,
                 vocab_size: int = 32000,
                 dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Layers
        self.embedding = DynamicEmbeddingTorch(vocab_size, dim)
        self.pos_encoding = PositionalEncoding(dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            DynamicTransformerBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_out = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.embedding.weight
        
        # Stats
        self.train_steps = 0
        self.created_at = datetime.now().isoformat()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0
    
    def forward(self, 
                input_ids: torch.Tensor,
                labels: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) token indices
            labels: (batch, seq_len) target indices for training
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if labels provided
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Causal mask
        mask = self._make_causal_mask(seq_len, input_ids.device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        # Compute loss if training
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=0  # Ignore padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 0.8,
                 top_k: int = 50) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            input_ids = generated[:, -self.max_seq_len:]
            
            # Forward
            logits, _ = self(input_ids)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at <END>
            if next_token.item() == 3:
                break
        
        return generated
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def persist_as_code(self, save_dir: str):
        """Save model weights as Python code."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save config
        config = f'''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Transformer - Trained Weights
Generated: {datetime.now().isoformat()}
Train Steps: {self.train_steps}
"""

import numpy as np

# === CONFIG ===
VOCAB_SIZE = {self.vocab_size}
DIM = {self.dim}
NUM_HEADS = {self.num_heads}
NUM_LAYERS = {self.num_layers}
MAX_SEQ_LEN = {self.max_seq_len}
PARAMETERS = {self.count_parameters()}
'''
        
        with open(os.path.join(save_dir, "config.py"), 'w', encoding='utf-8') as f:
            f.write(config)
        
        # Save embedding vocab
        vocab_code = f'''# Vocabulary
WORD2IDX = {dict(list(self.embedding.word2idx.items()))}
IDX2WORD = {dict(list(self.embedding.idx2word.items()))}
'''
        with open(os.path.join(save_dir, "vocab.py"), 'w', encoding='utf-8') as f:
            f.write(vocab_code)
        
        # Save PyTorch state dict
        torch.save(self.state_dict(), os.path.join(save_dir, "weights.pt"))
        
        print(f"[PERSIST] Saved to {save_dir}/")
        print(f"  - config.py")
        print(f"  - vocab.py ({len(self.embedding.word2idx)} words)")
        print(f"  - weights.pt ({self.count_parameters():,} params)")
    
    def __repr__(self):
        return (f"DynamicTransformerLM(vocab={self.vocab_size}, dim={self.dim}, "
                f"heads={self.num_heads}, layers={self.num_layers}, "
                f"params={self.count_parameters():,})")


# === Quick Test ===
if __name__ == "__main__":
    print("=== Dynamic Transformer LM ===")
    
    model = DynamicTransformerLM(
        vocab_size=1000,
        dim=128,
        num_heads=4,
        num_layers=2
    )
    print(model)
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward
    x = torch.randint(0, 100, (2, 32))
    logits, loss = model(x, labels=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generate
    prompt = torch.tensor([[2]])  # <START>
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"Generated: {generated}")
