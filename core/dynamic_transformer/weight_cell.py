# -*- coding: utf-8 -*-
"""
WeightCell: خلية الوزن الديناميكية
A neural network weight stored as executable Python code.

This is the fundamental building block of the Dynamic Transformer.
Instead of storing weights as binary tensors, we store them as code.
"""

import json
import math
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Union


class WeightCell:
    """
    خلية الوزن - تُخزن وزن شبكة عصبية ككود قابل للتنفيذ.
    
    Instead of:
        weight = torch.randn(512, 512)  # Binary tensor
    
    We have:
        weight = WeightCell(shape=(512, 512))  # Self-writing code
    """
    
    def __init__(self, 
                 shape: Tuple[int, ...],
                 name: str = None,
                 init_method: str = "xavier",
                 values: np.ndarray = None):
        """
        Initialize a weight cell.
        
        Args:
            shape: Shape of the weight matrix
            name: Name for identification
            init_method: "xavier", "normal", "zeros", "ones"
            values: Optional pre-computed values
        """
        self.shape = shape
        self.name = name or f"weight_{id(self)}"
        self.init_method = init_method
        self.learned_at = datetime.now().isoformat()
        self.update_count = 0
        
        # Initialize values
        if values is not None:
            self.values = np.array(values)
        else:
            self.values = self._initialize(shape, init_method)
    
    def _initialize(self, shape: Tuple[int, ...], method: str) -> np.ndarray:
        """Initialize weight values."""
        if method == "xavier":
            # Xavier/Glorot initialization
            fan_in = shape[0] if len(shape) > 0 else 1
            fan_out = shape[1] if len(shape) > 1 else 1
            std = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(*shape) * std
        
        elif method == "normal":
            return np.random.randn(*shape) * 0.02
        
        elif method == "zeros":
            return np.zeros(shape)
        
        elif method == "ones":
            return np.ones(shape)
        
        else:
            return np.random.randn(*shape) * 0.02
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Matrix multiplication: x @ W"""
        return np.dot(x, self.values)
    
    def update(self, delta: np.ndarray, learning_rate: float = 0.001):
        """Update weights with gradient."""
        self.values -= learning_rate * delta
        self.update_count += 1
        self.learned_at = datetime.now().isoformat()
    
    def to_source_code(self) -> str:
        """
        Serialize to executable Python code.
        This is the KEY feature: weights as code.
        """
        # Compact representation for large matrices
        if self.values.size > 100:
            # Store statistics + compressed form
            mean = float(self.values.mean())
            std = float(self.values.std())
            
            code = f'''
# --- WEIGHT CELL: {self.name} ---
{self.name} = WeightCell(
    shape={self.shape},
    name="{self.name}",
    init_method="from_stats"
)
{self.name}.values = np.random.randn{self.shape} * {std:.6f} + {mean:.6f}
{self.name}._load_compressed("{self._compress()}")
{self.name}.update_count = {self.update_count}
{self.name}.learned_at = "{self.learned_at}"
'''
        else:
            # Full values for small matrices
            values_str = np.array2string(self.values, separator=', ', 
                                         threshold=1000, max_line_width=200)
            code = f'''
# --- WEIGHT CELL: {self.name} ---
{self.name} = WeightCell(
    shape={self.shape},
    name="{self.name}",
    values=np.array({values_str})
)
{self.name}.update_count = {self.update_count}
{self.name}.learned_at = "{self.learned_at}"
'''
        return code
    
    def _compress(self) -> str:
        """Compress values to base64 for storage."""
        import base64
        import zlib
        
        # Convert to bytes and compress
        data = self.values.tobytes()
        compressed = zlib.compress(data, level=9)
        encoded = base64.b64encode(compressed).decode('ascii')
        
        return encoded
    
    def _load_compressed(self, encoded: str):
        """Load from compressed base64."""
        import base64
        import zlib
        
        compressed = base64.b64decode(encoded)
        data = zlib.decompress(compressed)
        self.values = np.frombuffer(data, dtype=np.float64).reshape(self.shape)
    
    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "name": self.name,
            "shape": self.shape,
            "init_method": self.init_method,
            "update_count": self.update_count,
            "learned_at": self.learned_at,
            "values_mean": float(self.values.mean()),
            "values_std": float(self.values.std())
        }
    
    def __repr__(self):
        return f"WeightCell('{self.name}', shape={self.shape}, updates={self.update_count})"
    
    def __matmul__(self, other):
        """Support @ operator: W @ x"""
        if isinstance(other, np.ndarray):
            return np.dot(self.values, other)
        elif isinstance(other, WeightCell):
            return np.dot(self.values, other.values)
        raise TypeError(f"Cannot matmul WeightCell with {type(other)}")


# === Quick Test ===
if __name__ == "__main__":
    # Create a weight
    w = WeightCell(shape=(4, 3), name="test_weight")
    print(w)
    print("Values:\n", w.values)
    print()
    
    # Forward pass
    x = np.random.randn(2, 4)
    y = w.forward(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print()
    
    # Update
    w.update(np.random.randn(4, 3) * 0.1)
    print("After update:", w)
    print()
    
    # Source code
    print("=== Source Code ===")
    print(w.to_source_code())
