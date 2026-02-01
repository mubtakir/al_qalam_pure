# -*- coding: utf-8 -*-
"""
GenerationEngine: محرك التوليد اللغوي
The core engine that generates text from vocabulary and patterns.
"""

import random
from typing import Dict, List, Optional, Tuple

from .word_cell import WordCell
from .pattern_cell import PatternCell
from .dynamic_vocab import DynamicVocab
from .dynamic_grammar import DynamicGrammar


class GenerationEngine:
    """
    محرك التوليد - يجمع المفردات والنحو لتوليد نص طبيعي.
    
    This is the heart of the Dynamic Language Engine:
    - Takes knowledge (concepts, facts) as input
    - Uses patterns and vocabulary to generate natural text
    - NO external LLM required
    """
    
    def __init__(self, vocab: DynamicVocab = None, grammar: DynamicGrammar = None):
        """Initialize with vocab and grammar managers."""
        self.vocab = vocab or DynamicVocab()
        self.grammar = grammar or DynamicGrammar()
        
        # Generation settings
        self.temperature = 0.7  # Randomness: 0=deterministic, 1=random
        self.max_length = 50  # Max words per generation
    
    def generate_from_pattern(self, pattern: PatternCell, context: Dict = None) -> str:
        """
        Generate a sentence from a pattern.
        
        Args:
            pattern: The pattern to use
            context: Optional context with constraints
                     e.g., {"subject": "القط", "style": "formal"}
        """
        context = context or {}
        words = []
        
        for i, pos in enumerate(pattern.structure):
            # Check if context provides a word for this slot
            if f"slot_{i}" in context:
                words.append(context[f"slot_{i}"])
                continue
            
            # Get candidates from pattern's slot history
            candidates = pattern.get_best_candidates(i, n=10)
            
            if candidates:
                # Choose based on temperature
                if self.temperature < 0.1:
                    # Deterministic: pick most frequent
                    word_id = candidates[0]
                else:
                    # Random: weighted by position in list
                    weights = [1.0 / (j + 1) for j in range(len(candidates))]
                    word_id = random.choices(candidates, weights=weights)[0]
                
                # Extract actual word from word_id
                word = word_id.replace("word_", "")
                words.append(word)
            else:
                # Fallback: find any word with this POS
                pos_words = self.vocab.find_by_pos(pos)
                if pos_words:
                    word = random.choice(pos_words).word
                    words.append(word)
                else:
                    words.append(f"[{pos}]")  # Placeholder
        
        return pattern.generate(words)
    
    def generate_simple(self, 
                        subject: str, 
                        predicate: str = None, 
                        obj: str = None) -> str:
        """
        Generate a simple sentence with subject, verb, object.
        
        Args:
            subject: The subject (e.g., "القط")
            predicate: The verb/action (e.g., "يأكل")
            obj: The object (optional, e.g., "السمك")
        """
        if obj:
            # NOUN VERB NOUN pattern
            pattern = self.grammar.find_pattern(["NOUN", "VERB", "NOUN"])
            if pattern:
                return pattern.generate([subject, predicate, obj])
            return f"{subject} {predicate} {obj}"
        elif predicate:
            # NOUN VERB pattern
            pattern = self.grammar.find_pattern(["NOUN", "VERB"])
            if pattern:
                return pattern.generate([subject, predicate])
            return f"{subject} {predicate}"
        else:
            # Just the subject
            return subject
    
    def generate_description(self, entity: str, attributes: List[Tuple[str, str]]) -> str:
        """
        Generate a description of an entity.
        
        Args:
            entity: The thing to describe (e.g., "القط")
            attributes: List of (attribute, value) tuples
                       e.g., [("لون", "أسود"), ("حجم", "كبير")]
        """
        sentences = []
        
        for attr, value in attributes:
            # Try NOUN ADJ pattern
            pattern = self.grammar.find_pattern(["NOUN", "ADJ"])
            if pattern and attr == "":
                sentences.append(pattern.generate([entity, value]))
            else:
                sentences.append(f"{entity} {attr} {value}")
        
        return "، ".join(sentences) + "."
    
    def generate_from_knowledge(self, 
                                concept_cell, 
                                include_facts: bool = True) -> str:
        """
        Generate text describing a knowledge cell.
        
        Args:
            concept_cell: A DynamicCell with knowledge
            include_facts: Whether to include related facts
        """
        parts = []
        
        name = concept_cell.metadata.get("name", concept_cell.id)
        cell_type = concept_cell.type
        
        # Opening
        if cell_type == "concept":
            parts.append(f"{name} هو مفهوم")
        elif cell_type == "instance":
            parts.append(f"{name}")
        
        # Facts
        if include_facts and "facts" in concept_cell.metadata:
            facts = concept_cell.metadata["facts"]
            for fact in facts[:3]:  # Limit to 3 facts
                parts.append(fact)
        
        return "، ".join(parts) + "."
    
    def learn_and_generate(self, input_text: str) -> str:
        """
        Learn from input text, then generate a similar sentence.
        This is the self-improving loop.
        """
        # 1. Learn from input
        self.vocab.learn_from_text(input_text)
        
        # 2. Simple tokenize
        import re
        words = re.findall(r'[\u0600-\u06FF]+', input_text)
        
        if len(words) < 2:
            return input_text  # Too short to learn pattern
        
        # 3. Guess POS and learn pattern
        tags = [self.vocab.get_word(w).pos if self.vocab.get_word(w) else "NOUN" 
                for w in words]
        
        pattern = self.grammar.learn_from_tagged(words, tags, input_text)
        
        # 4. Generate new sentence from learned pattern
        if pattern:
            return self.generate_from_pattern(pattern)
        
        return input_text
    
    def __repr__(self):
        return f"GenerationEngine(vocab={len(self.vocab)}, patterns={len(self.grammar)})"


# === Quick Test ===
if __name__ == "__main__":
    engine = GenerationEngine()
    
    # Learn some vocabulary
    engine.vocab.add_word("القط", pos="NOUN")
    engine.vocab.add_word("الكلب", pos="NOUN")
    engine.vocab.add_word("السمك", pos="NOUN")
    engine.vocab.add_word("يأكل", pos="VERB")
    engine.vocab.add_word("يشرب", pos="VERB")
    engine.vocab.add_word("كبير", pos="ADJ")
    engine.vocab.add_word("صغير", pos="ADJ")
    
    # Learn patterns
    engine.grammar.learn_from_tagged(
        ["القط", "يأكل", "السمك"],
        ["NOUN", "VERB", "NOUN"],
        "القط يأكل السمك"
    )
    engine.grammar.learn_from_tagged(
        ["الكلب", "يشرب", "الماء"],
        ["NOUN", "VERB", "NOUN"],
        "الكلب يشرب الماء"
    )
    
    # Generate
    print("Engine:", engine)
    print("Simple:", engine.generate_simple("القط", "يحب", "اللعب"))
    
    # Test pattern generation
    pattern = engine.grammar.get_best_pattern(["NOUN", "VERB", "NOUN"])
    if pattern:
        print("From pattern:", engine.generate_from_pattern(pattern))
