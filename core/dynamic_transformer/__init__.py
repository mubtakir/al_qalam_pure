# -*- coding: utf-8 -*-
"""
Dynamic Transformer Package
حزمة الـ Transformer الديناميكي

Al-Qalam V4.0 - A self-writing Transformer
"""

from .weight_cell import WeightCell
from .dynamic_embedding import DynamicEmbedding
from .dynamic_attention import DynamicAttention
from .dynamic_ffn import DynamicFFN, DynamicLayerNorm
from .dynamic_transformer import DynamicTransformer, DynamicTransformerBlock

__all__ = [
    "WeightCell",
    "DynamicEmbedding",
    "DynamicAttention",
    "DynamicFFN",
    "DynamicLayerNorm",
    "DynamicTransformer",
    "DynamicTransformerBlock"
]

__version__ = "0.1.0"
