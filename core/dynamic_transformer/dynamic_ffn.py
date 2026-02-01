# -*- coding: utf-8 -*-
"""
DynamicFFN: Feed-Forward Network ديناميكي
A feed-forward network that can expand dynamically.
"""

import numpy as np
from datetime import datetime
from typing import List, Optional

from .weight_cell import WeightCell


class DynamicFFN:
    """
    FFN ديناميكي - يمكن أن يتوسع ديناميكياً.
    
    Standard FFN: Linear -> ReLU -> Linear
    Dynamic FFN: Same, but weights stored as code.
    """
    
    def __init__(self,
                 dim: int = 256,
                 hidden_dim: int = None,
                 name: str = "ffn"):
        """
        Initialize dynamic FFN.
        
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: 4 * dim)
            name: Name for identification
        """
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
        self.name = name
        
        # Weights
        self.W1 = WeightCell(
            shape=(dim, self.hidden_dim),
            name=f"{name}_W1",
            init_method="xavier"
        )
        self.b1 = WeightCell(
            shape=(self.hidden_dim,),
            name=f"{name}_b1",
            init_method="zeros"
        )
        self.W2 = WeightCell(
            shape=(self.hidden_dim, dim),
            name=f"{name}_W2",
            init_method="xavier"
        )
        self.b2 = WeightCell(
            shape=(dim,),
            name=f"{name}_b2",
            init_method="zeros"
        )
        
        # Statistics
        self.forward_count = 0
        self.learned_at = datetime.now().isoformat()
    
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray, activation: str = "gelu") -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input (seq_len, dim) or (batch, seq_len, dim)
            activation: "gelu" or "relu"
        """
        # First linear
        hidden = x @ self.W1.values + self.b1.values
        
        # Activation
        if activation == "gelu":
            hidden = self._gelu(hidden)
        else:
            hidden = self._relu(hidden)
        
        # Second linear
        output = hidden @ self.W2.values + self.b2.values
        
        self.forward_count += 1
        return output
    
    def to_source_code(self) -> str:
        """Serialize to Python code."""
        code = f'''
# === DYNAMIC FFN: {self.name} ===
{self.name} = DynamicFFN(
    dim={self.dim},
    hidden_dim={self.hidden_dim},
    name="{self.name}"
)
{self.name}.forward_count = {self.forward_count}

{self.W1.to_source_code()}
{self.b1.to_source_code()}
{self.W2.to_source_code()}
{self.b2.to_source_code()}
'''
        return code
    
    def __repr__(self):
        return f"DynamicFFN(dim={self.dim}, hidden={self.hidden_dim})"


class DynamicLayerNorm:
    """Layer Normalization with dynamic parameters."""
    
    def __init__(self, dim: int, name: str = "ln", eps: float = 1e-6):
        self.dim = dim
        self.name = name
        self.eps = eps
        
        self.gamma = WeightCell(shape=(dim,), name=f"{name}_gamma", init_method="ones")
        self.beta = WeightCell(shape=(dim,), name=f"{name}_beta", init_method="zeros")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma.values * normalized + self.beta.values
    
    def __repr__(self):
        return f"DynamicLayerNorm(dim={self.dim})"


# === Quick Test ===
if __name__ == "__main__":
    ffn = DynamicFFN(dim=64)
    ln = DynamicLayerNorm(dim=64)
    
    print(ffn)
    print(ln)
    
    x = np.random.randn(5, 64)
    
    # LayerNorm
    x_norm = ln.forward(x)
    print("After LayerNorm - mean:", x_norm.mean(), "std:", x_norm.std())
    
    # FFN
    output = ffn.forward(x_norm)
    print("FFN output shape:", output.shape)
