# -*- coding: utf-8 -*-
"""
DynamicGrammar: مدير القواعد النحوية الديناميكي
Manages sentence patterns that grow and persist as code.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

from .pattern_cell import PatternCell


class DynamicGrammar:
    """
    مدير النحو الديناميكي - يتعلم أنماط الجمل ويحفظها ككود.
    
    Features:
    - Extract patterns from sentences
    - Find matching patterns for generation
    - Persist patterns as Python code
    """
    
    def __init__(self, persist_path: str = None):
        """
        Initialize the grammar manager.
        
        Args:
            persist_path: Path to save patterns.py
        """
        self.patterns: Dict[str, PatternCell] = {}  # pattern_id -> PatternCell
        self.persist_path = persist_path
        self._pattern_counter = 0
    
    def _generate_pattern_id(self, structure: List[str]) -> str:
        """Generate a unique pattern ID."""
        self._pattern_counter += 1
        struct_abbrev = "_".join([s[:1].lower() for s in structure])
        return f"{struct_abbrev}_{self._pattern_counter}"
    
    def add_pattern(self,
                    structure: List[str],
                    examples: List[str] = None,
                    template: str = None) -> PatternCell:
        """Add a new pattern or update existing one."""
        
        # Check if pattern already exists
        for p in self.patterns.values():
            if p.structure == structure:
                if examples:
                    p.examples.extend([e for e in examples if e not in p.examples])
                p.frequency += 1
                return p
        
        # Create new pattern
        pattern_id = self._generate_pattern_id(structure)
        pattern = PatternCell(
            pattern_id=pattern_id,
            structure=structure,
            examples=examples,
            template=template
        )
        
        self.patterns[pattern.id] = pattern
        return pattern
    
    def learn_from_tagged(self, words: List[str], tags: List[str], sentence: str):
        """
        Learn a pattern from a POS-tagged sentence.
        
        Args:
            words: List of words
            tags: List of POS tags
            sentence: The original sentence
        """
        if len(words) != len(tags):
            return None
        
        pattern = self.add_pattern(structure=tags, examples=[sentence])
        pattern.record_example(words)
        return pattern
    
    def learn_from_sentence(self, sentence: str, word_to_pos: Dict[str, str]):
        """
        Learn a pattern from a sentence using a word->POS mapping.
        """
        # Simple Arabic tokenization
        words = re.findall(r'[\u0600-\u06FF]+', sentence)
        tags = [word_to_pos.get(w, "NOUN") for w in words]
        
        return self.learn_from_tagged(words, tags, sentence)
    
    def find_pattern(self, structure: List[str]) -> Optional[PatternCell]:
        """Find a pattern matching the given structure."""
        for p in self.patterns.values():
            if p.structure == structure:
                return p
        return None
    
    def find_patterns_by_length(self, length: int) -> List[PatternCell]:
        """Find all patterns of a specific length."""
        return [p for p in self.patterns.values() if len(p.structure) == length]
    
    def get_best_pattern(self, structure: List[str] = None, length: int = None) -> Optional[PatternCell]:
        """Get the best pattern based on frequency and success rate."""
        candidates = list(self.patterns.values())
        
        if structure:
            candidates = [p for p in candidates if p.structure == structure]
        elif length:
            candidates = [p for p in candidates if len(p.structure) == length]
        
        if not candidates:
            return None
        
        # Score by frequency * success_rate
        return max(candidates, key=lambda p: p.frequency * p.success_rate)
    
    def generate_sentence(self, pattern_id: str, words: List[str]) -> str:
        """Generate a sentence using a pattern."""
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            raise ValueError(f"Pattern {pattern_id} not found")
        
        return pattern.generate(words)
    
    def to_source_code(self) -> str:
        """Generate all patterns as executable Python code."""
        header = '''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Grammar Patterns
Auto-generated - DO NOT EDIT MANUALLY
Generated: {timestamp}
Total Patterns: {count}
"""

from core.language_engine.pattern_cell import PatternCell

# === PATTERNS ===
'''.format(timestamp=datetime.now().isoformat(), count=len(self.patterns))
        
        # Sort by frequency
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )
        
        code = header
        for pattern in sorted_patterns:
            code += pattern.to_source_code()
            code += "\n"
        
        return code
    
    def persist(self) -> bool:
        """Save patterns to file."""
        if not self.persist_path:
            return False
        
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        code = self.to_source_code()
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return True
    
    def __len__(self):
        return len(self.patterns)
    
    def __repr__(self):
        return f"DynamicGrammar({len(self.patterns)} patterns)"


# === Quick Test ===
if __name__ == "__main__":
    grammar = DynamicGrammar()
    
    # Learn patterns
    grammar.learn_from_tagged(
        words=["القط", "يأكل", "السمك"],
        tags=["NOUN", "VERB", "NOUN"],
        sentence="القط يأكل السمك"
    )
    
    grammar.learn_from_tagged(
        words=["الطفل", "يقرأ", "الكتاب"],
        tags=["NOUN", "VERB", "NOUN"],
        sentence="الطفل يقرأ الكتاب"
    )
    
    grammar.learn_from_tagged(
        words=["الشمس", "ساطعة"],
        tags=["NOUN", "ADJ"],
        sentence="الشمس ساطعة"
    )
    
    print(grammar)
    print("Best NOUN-VERB-NOUN pattern:", grammar.get_best_pattern(["NOUN", "VERB", "NOUN"]))
    print()
    print(grammar.to_source_code())
