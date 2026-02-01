# -*- coding: utf-8 -*-
"""
Pure Dynamic Transformer - Weights as Python Code ONLY
Transformer ديناميكي صافي - الأوزان ككود Python فقط

NO .pt FILES! True to Al-Qalam philosophy.
"""

import os
import math
import base64
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json


def compress_array(arr: np.ndarray) -> str:
    """Compress numpy array to base64 string."""
    data = arr.astype(np.float32).tobytes()
    compressed = zlib.compress(data, level=9)
    return base64.b64encode(compressed).decode('ascii')


def decompress_array(encoded: str, shape: Tuple[int, ...]) -> np.ndarray:
    """Decompress base64 string to numpy array."""
    compressed = base64.b64decode(encoded)
    data = zlib.decompress(compressed)
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


class PureDynamicTransformer(nn.Module):
    """
    Transformer ديناميكي صافي - يحفظ أوزانه ككود Python فقط.
    
    NO .pt files!
    All weights saved as compressed Python code.
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
        self.head_dim = dim // num_heads
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        
        # Transformer blocks (simplified for code serialization)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'qkv': nn.Linear(dim, 3 * dim),
                'out': nn.Linear(dim, dim),
                'ln2': nn.LayerNorm(dim),
                'fc1': nn.Linear(dim, 4 * dim),
                'fc2': nn.Linear(4 * dim, dim),
            }))
        
        self.ln_out = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Stats
        self.train_steps = 0
        self.created_at = datetime.now().isoformat()
        
        # V5.0 Semantic Knowledge base
        self.semantic_knowledge: Dict[str, Any] = {}
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Transformer blocks
        for block in self.blocks:
            # Self-attention
            residual = x
            x = block['ln1'](x)
            
            qkv = block['qkv'](x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
            x = residual + block['out'](out)
            
            # FFN
            residual = x
            x = block['ln2'](x)
            x = residual + block['fc2'](F.gelu(block['fc1'](x)))
        
        x = self.ln_out(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=0
            )
        
        return logits, loss

    def _analyze_semantic_relations(self, word2idx: Dict[str, int]) -> Dict[str, Any]:
        """
        V5.0: Analyze Attention weights to find strong semantic relations.
        Maps neuron clusters to linguistic patterns.
        """
        idx2word = {v: k for k, v in word2idx.items()}
        knowledge = {}
        
        # Simplified semantic mapping: Find top words by embedding variance
        with torch.no_grad():
            emb = self.embedding.weight.cpu().numpy()
            # Variance indicates importance/distinctiveness
            variance = np.var(emb, axis=1)
            top_indices = np.argsort(variance)[-50:] # Top 50 semantic hubs
            
            hubs = [idx2word.get(idx, f"token_{idx}") for idx in top_indices]
            knowledge["semantic_hubs"] = hubs
            knowledge["analysis_date"] = datetime.now().isoformat()
            
        return knowledge

    def update_vocab(self, new_vocab_size: int):
        """
        V5.0: Dynamically expand embedding and LM head without losing knowledge.
        """
        if new_vocab_size <= self.vocab_size:
            return
            
        print(f"[V5.0] Expanding Vocab: {self.vocab_size} -> {new_vocab_size}")
        
        # Old layers
        old_emb = self.embedding
        
        # New layers
        self.embedding = nn.Embedding(new_vocab_size, self.dim).to(old_emb.weight.device)
        self.lm_head = nn.Linear(self.dim, new_vocab_size, bias=False).to(old_emb.weight.device)
        
        # Copy old weights
        with torch.no_grad():
            self.embedding.weight[:self.vocab_size] = old_emb.weight
            # Re-tie weights
            self.lm_head.weight = self.embedding.weight
            
        self.vocab_size = new_vocab_size

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50, 
                 temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
        self.eval()
        generated = prompt_ids.clone()
        
        for _ in range(max_new_tokens):
            input_ids = generated[:, -self.max_seq_len:]
            logits, _ = self(input_ids)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == 3:  # <END>
                break
        
        return generated
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def to_python_code(self, save_dir: str, word2idx: Dict[str, int] = None):
        """
        Save ENTIRE model as Python code - NO .pt files!
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # === 1. Config ===
        config_code = f'''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Transformer - Pure Code Weights
Generated: {datetime.now().isoformat()}
Train Steps: {self.train_steps}

NO .pt FILES - TRUE DYNAMIC MODEL
"""

# === CONFIG ===
VOCAB_SIZE = {self.vocab_size}
DIM = {self.dim}
NUM_HEADS = {self.num_heads}
NUM_LAYERS = {self.num_layers}
MAX_SEQ_LEN = {self.max_seq_len}
PARAMETERS = {self.count_parameters()}
TRAIN_STEPS = {self.train_steps}
VERSION = "5.0"
'''
        
        with open(os.path.join(save_dir, "config.py"), 'w', encoding='utf-8') as f:
            f.write(config_code)
            
        # === V5.0 Knowledge Map ===
        if word2idx:
            knowledge = self._analyze_semantic_relations(word2idx)
            knowledge_code = f'''# -*- coding: utf-8 -*-
"""
Al-Qalam V5.0 Knowledge Map
Extracted Semantic Relations
"""

SEMANTIC_METADATA = {json.dumps(knowledge, indent=4, ensure_ascii=False)}
'''
            with open(os.path.join(save_dir, "knowledge_map.py"), 'w', encoding='utf-8') as f:
                f.write(knowledge_code)
        
        # === 2. Vocabulary ===
        if word2idx:
            vocab_code = f'''# -*- coding: utf-8 -*-
"""Vocabulary - {len(word2idx)} words"""

WORD2IDX = {repr(word2idx)}

IDX2WORD = {{v: k for k, v in WORD2IDX.items()}}
'''
            with open(os.path.join(save_dir, "vocab.py"), 'w', encoding='utf-8') as f:
                f.write(vocab_code)
        
        # === 3. Weights as compressed Python ===
        weights_code = '''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Transformer - Weights as Code
All weights compressed and stored as Python strings.

To load: Use decompress_array(WEIGHTS["name"], shape)
"""

import base64
import zlib
import numpy as np

def decompress_array(encoded: str, shape: tuple) -> np.ndarray:
    """Decompress base64 string to numpy array."""
    compressed = base64.b64decode(encoded)
    data = zlib.decompress(compressed)
    return np.frombuffer(data, dtype=np.float32).reshape(shape)

# === WEIGHTS ===
WEIGHTS = {
'''
        
        # Save each parameter
        for name, param in self.named_parameters():
            arr = param.detach().cpu().numpy()
            compressed = compress_array(arr)
            shape = arr.shape
            weights_code += f'    "{name}": ("{compressed}", {shape}),\n'
        
        weights_code += '''}

def load_weights():
    """Load all weights as numpy arrays."""
    return {name: decompress_array(data, shape) for name, (data, shape) in WEIGHTS.items()}
'''
        
        with open(os.path.join(save_dir, "weights.py"), 'w', encoding='utf-8') as f:
            f.write(weights_code)
        
        # === 4. Loader ===
        loader_code = '''# -*- coding: utf-8 -*-
"""
Model Loader - Load from Python code (NO .pt required!)
"""

import numpy as np
import torch

from .config import *
from .vocab import WORD2IDX, IDX2WORD
from .weights import load_weights

def load_model():
    """Load model from Python code weights."""
    from core.dynamic_transformer.pure_dynamic_transformer import PureDynamicTransformer
    
    # Create model
    model = PureDynamicTransformer(
        vocab_size=VOCAB_SIZE,
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # Load weights
    weights = load_weights()
    state_dict = model.state_dict()
    
    for name, arr in weights.items():
        if name in state_dict:
            state_dict[name] = torch.from_numpy(arr)
    
    model.load_state_dict(state_dict)
    model.train_steps = TRAIN_STEPS
    
    return model, WORD2IDX, IDX2WORD

if __name__ == "__main__":
    model, word2idx, idx2word = load_model()
    print(f"Loaded: {model.count_parameters():,} params")
'''
        
        with open(os.path.join(save_dir, "loader.py"), 'w', encoding='utf-8') as f:
            f.write(loader_code)
        
        # Init
        with open(os.path.join(save_dir, "__init__.py"), 'w', encoding='utf-8') as f:
            f.write("from .loader import load_model\n")
        
        # Stats
        weights_file = os.path.join(save_dir, "weights.py")
        size_mb = os.path.getsize(weights_file) / (1024 * 1024)
        
        print(f"\n[SAVED AS PYTHON CODE]")
        print(f"  Directory: {save_dir}/")
        print(f"  - config.py")
        print(f"  - vocab.py ({len(word2idx) if word2idx else 0} words)")
        print(f"  - weights.py ({size_mb:.1f} MB)")
        print(f"  - loader.py")
        print(f"  NO .pt FILES! ✅")


# === Quick Test ===
if __name__ == "__main__":
    print("=== Pure Dynamic Transformer ===")
    
    model = PureDynamicTransformer(
        vocab_size=1000,
        dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=64
    )
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward
    x = torch.randint(0, 100, (2, 32))
    logits, loss = model(x, labels=x)
    print(f"Loss: {loss.item():.4f}")
    
    # Save as code
    model.to_python_code("test_export", {"test": 0, "word": 1})
