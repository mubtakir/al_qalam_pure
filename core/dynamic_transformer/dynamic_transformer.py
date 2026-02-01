# -*- coding: utf-8 -*-
"""
DynamicTransformer: التجميع الكامل
The complete Dynamic Transformer that stores itself as code.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .weight_cell import WeightCell
from .dynamic_embedding import DynamicEmbedding
from .dynamic_attention import DynamicAttention
from .dynamic_ffn import DynamicFFN, DynamicLayerNorm


class DynamicTransformerBlock:
    """Single Transformer block with dynamic components."""
    
    def __init__(self, dim: int, num_heads: int, block_id: int = 0):
        self.dim = dim
        self.num_heads = num_heads
        self.block_id = block_id
        
        # Components
        self.ln1 = DynamicLayerNorm(dim, name=f"block{block_id}_ln1")
        self.attention = DynamicAttention(dim, num_heads, name=f"block{block_id}_attn")
        self.ln2 = DynamicLayerNorm(dim, name=f"block{block_id}_ln2")
        self.ffn = DynamicFFN(dim, name=f"block{block_id}_ffn")
    
    def forward(self, x: np.ndarray, words: List[str] = None) -> np.ndarray:
        """Forward pass with residual connections."""
        # Self-attention
        residual = x
        x = self.ln1.forward(x)
        x, _ = self.attention.forward(x, words=words)
        x = x + residual
        
        # FFN
        residual = x
        x = self.ln2.forward(x)
        x = self.ffn.forward(x)
        x = x + residual
        
        return x


class DynamicTransformer:
    """
    Transformer ديناميكي كامل - يحفظ نفسه ككود Python.
    
    Architecture:
    - DynamicEmbedding (vocab, positions)
    - N x DynamicTransformerBlock
    - Output projection
    
    All weights stored as Python code, not binary tensors.
    """
    
    def __init__(self,
                 dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 max_seq_len: int = 512,
                 base_dir: str = "."):
        """
        Initialize Dynamic Transformer.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            max_seq_len: Maximum sequence length
            base_dir: Base directory for saving
        """
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.base_dir = base_dir
        
        # Paths
        self.save_dir = os.path.join(base_dir, "vault", "transformer")
        
        # Token embedding
        self.embedding = DynamicEmbedding(
            dim=dim,
            name="token_embedding",
            persist_path=os.path.join(self.save_dir, "embeddings.py")
        )
        
        # Positional embedding (learnable)
        self.pos_embedding = WeightCell(
            shape=(max_seq_len, dim),
            name="pos_embedding",
            init_method="normal"
        )
        
        # Transformer blocks
        self.blocks: List[DynamicTransformerBlock] = []
        for i in range(num_layers):
            self.blocks.append(DynamicTransformerBlock(dim, num_heads, i))
        
        # Output
        self.ln_out = DynamicLayerNorm(dim, name="ln_out")
        self.output_proj = WeightCell(
            shape=(dim, dim),
            name="output_proj",
            init_method="xavier"
        )
        
        # Statistics
        self.forward_count = 0
        self.train_steps = 0
        self.created_at = datetime.now().isoformat()
    
    def forward(self, 
                words: List[str],
                return_hidden: bool = False) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            words: List of words (tokens)
            return_hidden: Return hidden states instead of logits
        """
        seq_len = len(words)
        
        # Token embeddings
        x = self.embedding.embed_sentence(words)  # (seq_len, dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding.values[:seq_len]
        
        # Through transformer blocks
        for block in self.blocks:
            x = block.forward(x, words=words)
        
        # Final layer norm
        x = self.ln_out.forward(x)
        
        self.forward_count += 1
        
        if return_hidden:
            return x
        
        # Project to output
        output = x @ self.output_proj.values
        return output
    
    def generate(self, 
                 prompt: List[str],
                 max_tokens: int = 20,
                 temperature: float = 1.0) -> List[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Starting words
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        generated = list(prompt)
        
        for _ in range(max_tokens):
            # Forward pass
            hidden = self.forward(generated, return_hidden=True)
            
            # Get last token hidden state
            last_hidden = hidden[-1]  # (dim,)
            
            # Find most similar word in vocabulary
            best_word = None
            best_score = -float('inf')
            
            for word, cell in self.embedding.embeddings.items():
                if word.startswith("<"):  # Skip special tokens
                    continue
                
                score = np.dot(last_hidden, cell.values) / (
                    np.linalg.norm(last_hidden) * np.linalg.norm(cell.values) + 1e-8
                )
                
                # Add temperature
                score = score / temperature
                
                if score > best_score:
                    best_score = score
                    best_word = word
            
            if best_word is None or best_word == "<END>":
                break
            
            generated.append(best_word)
        
        return generated
    
    def learn_from_sentence(self, 
                            words: List[str],
                            learning_rate: float = 0.001):
        """
        Simple learning: update embeddings based on context.
        """
        # Add all words to vocabulary
        for word in words:
            self.embedding.add_word(word)
        
        # Learn embeddings from context
        for i, word in enumerate(words):
            context = words[max(0, i-2):i] + words[i+1:min(len(words), i+3)]
            if context:
                self.embedding.learn_from_context(word, context, learning_rate)
        
        # Forward pass to learn attention patterns
        self.forward(words)
        
        self.train_steps += 1
    
    def persist(self):
        """Save entire model to Python files."""
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save embeddings
        self.embedding.persist()
        
        # Save transformer config and weights
        config_code = f'''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Transformer
Auto-generated - DO NOT EDIT MANUALLY
Generated: {datetime.now().isoformat()}
"""

# === CONFIG ===
DIM = {self.dim}
NUM_HEADS = {self.num_heads}
NUM_LAYERS = {self.num_layers}
MAX_SEQ_LEN = {self.max_seq_len}
TRAIN_STEPS = {self.train_steps}
FORWARD_COUNT = {self.forward_count}

# Positional Embedding
{self.pos_embedding.to_source_code()}
'''
        
        with open(os.path.join(self.save_dir, "config.py"), 'w', encoding='utf-8') as f:
            f.write(config_code)
        
        print(f"[PERSIST] Saved to {self.save_dir}/")
        print(f"  - embeddings.py ({len(self.embedding)} words)")
        print(f"  - config.py")
    
    def __repr__(self):
        return (f"DynamicTransformer(dim={self.dim}, heads={self.num_heads}, "
                f"layers={self.num_layers}, vocab={len(self.embedding)})")


# === Quick Test ===
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=== Creating Dynamic Transformer ===")
    model = DynamicTransformer(dim=64, num_heads=4, num_layers=2, base_dir=".")
    print(model)
    print()
    
    # Learn from sentences
    sentences = [
        ["القط", "يأكل", "السمك"],
        ["الكلب", "يشرب", "الماء"],
        ["الطفل", "يقرأ", "الكتاب"],
    ]
    
    print("=== Learning ===")
    for sentence in sentences:
        model.learn_from_sentence(sentence)
        print(f"Learned: {' '.join(sentence)}")
    
    print()
    print(f"Vocabulary: {len(model.embedding)} words")
    print(f"Train steps: {model.train_steps}")
    
    # Generate
    print()
    print("=== Generating ===")
    prompt = ["القط"]
    generated = model.generate(prompt, max_tokens=3)
    print(f"Prompt: {prompt}")
    print(f"Generated: {' '.join(generated)}")
    
    # Save
    print()
    model.persist()
