# -*- coding: utf-8 -*-
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
