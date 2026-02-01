# -*- coding: utf-8 -*-
"""
DynamicAttention: Attention ديناميكي
Self-attention that learns and stores patterns as code.

Key Innovation: Attention patterns stored as Python code, not just computed.
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .weight_cell import WeightCell


class DynamicAttention:
    """
    Attention ديناميكي - يتعلم ويحفظ أنماط الانتباه ككود.
    
    Unlike standard attention:
    - Weights (Q, K, V) stored as WeightCells
    - Learned attention patterns are persisted
    - Can add new heads dynamically
    """
    
    def __init__(self,
                 dim: int = 256,
                 num_heads: int = 4,
                 name: str = "attention"):
        """
        Initialize dynamic attention.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            name: Name for identification
        """
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.name = name
        
        # Weight matrices as WeightCells
        self.W_q = WeightCell(shape=(dim, dim), name=f"{name}_Wq", init_method="xavier")
        self.W_k = WeightCell(shape=(dim, dim), name=f"{name}_Wk", init_method="xavier")
        self.W_v = WeightCell(shape=(dim, dim), name=f"{name}_Wv", init_method="xavier")
        self.W_o = WeightCell(shape=(dim, dim), name=f"{name}_Wo", init_method="xavier")
        
        # Learned attention patterns (word-pair -> attention strength)
        self.learned_patterns: Dict[Tuple[str, str], float] = {}
        
        # Statistics
        self.forward_count = 0
        self.learned_at = datetime.now().isoformat()
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple heads: (batch, seq, dim) -> (batch, heads, seq, head_dim)"""
        batch_size = x.shape[0] if len(x.shape) == 3 else 1
        seq_len = x.shape[-2]
        
        if len(x.shape) == 2:
            x = x.reshape(1, seq_len, self.dim)
        
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)
    
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back: (batch, heads, seq, head_dim) -> (batch, seq, dim)"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.dim)
    
    def forward(self, 
                x: np.ndarray, 
                mask: np.ndarray = None,
                words: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through attention.
        
        Args:
            x: Input (seq_len, dim) or (batch, seq_len, dim)
            mask: Optional attention mask
            words: Optional word list for learning patterns
        
        Returns:
            output: Attended output
            attention_weights: Attention weights
        """
        # Ensure 3D input
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_q.values  # (batch, seq, dim)
        K = x @ self.W_k.values
        V = x @ self.W_v.values
        
        # Split into heads
        Q = self._split_heads(Q)  # (batch, heads, seq, head_dim)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Scaled dot-product attention
        scale = np.sqrt(self.head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale  # (batch, heads, seq, seq)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * (-1e9)
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)  # (batch, heads, seq, head_dim)
        
        # Merge heads
        context = self._merge_heads(context)  # (batch, seq, dim)
        
        # Final projection
        output = context @ self.W_o.values
        
        # Learn patterns from words
        if words is not None:
            self._learn_patterns(attention_weights[0, 0], words)
        
        self.forward_count += 1
        
        # Remove batch dim if input was 2D
        if batch_size == 1:
            output = output.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
        
        return output, attention_weights
    
    def _learn_patterns(self, attention: np.ndarray, words: List[str]):
        """Learn attention patterns from a forward pass."""
        seq_len = len(words)
        
        for i in range(min(seq_len, attention.shape[0])):
            for j in range(min(seq_len, attention.shape[1])):
                if attention[i, j] > 0.1:  # Significant attention
                    key = (words[i], words[j])
                    
                    if key not in self.learned_patterns:
                        self.learned_patterns[key] = attention[i, j]
                    else:
                        # Running average
                        self.learned_patterns[key] = 0.9 * self.learned_patterns[key] + 0.1 * attention[i, j]
    
    def get_attention_bias(self, word1: str, word2: str) -> float:
        """Get learned attention bias between two words."""
        key = (word1, word2)
        return self.learned_patterns.get(key, 0.0)
    
    def to_source_code(self) -> str:
        """Serialize attention layer to Python code."""
        code = f'''
# === DYNAMIC ATTENTION: {self.name} ===
{self.name} = DynamicAttention(
    dim={self.dim},
    num_heads={self.num_heads},
    name="{self.name}"
)
{self.name}.forward_count = {self.forward_count}
{self.name}.learned_at = "{self.learned_at}"

# Weights
{self.W_q.to_source_code()}
{self.W_k.to_source_code()}
{self.W_v.to_source_code()}
{self.W_o.to_source_code()}

# Learned Patterns (top 100)
{self.name}.learned_patterns = {{
'''
        # Top 100 patterns
        sorted_patterns = sorted(
            self.learned_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:100]
        
        for (w1, w2), score in sorted_patterns:
            code += f'    ("{w1}", "{w2}"): {score:.4f},\n'
        
        code += "}\n"
        return code
    
    def __repr__(self):
        return f"DynamicAttention(dim={self.dim}, heads={self.num_heads}, patterns={len(self.learned_patterns)})"


# === Quick Test ===
if __name__ == "__main__":
    attn = DynamicAttention(dim=64, num_heads=4)
    print(attn)
    
    # Create input
    seq_len = 5
    x = np.random.randn(seq_len, 64)
    words = ["القط", "الأسود", "يأكل", "السمك", "الطازج"]
    
    # Forward
    output, weights = attn.forward(x, words=words)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention shape:", weights.shape)
    print()
    
    # Check learned patterns
    print("Learned patterns:")
    for (w1, w2), score in list(attn.learned_patterns.items())[:5]:
        print(f"  {w1} -> {w2}: {score:.3f}")
