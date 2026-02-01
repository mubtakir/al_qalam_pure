# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Language Engine V3.0
المحرك اللغوي الديناميكي

A self-writing language engine that stores vocabulary and patterns as code.
"""

from .word_cell import WordCell
from .pattern_cell import PatternCell
from .dynamic_vocab import DynamicVocab
from .dynamic_grammar import DynamicGrammar
from .generation_engine import GenerationEngine

__all__ = [
    "WordCell",
    "PatternCell", 
    "DynamicVocab",
    "DynamicGrammar",
    "GenerationEngine"
]

__version__ = "3.0.0"
