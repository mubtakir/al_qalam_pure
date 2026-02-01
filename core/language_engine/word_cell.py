# -*- coding: utf-8 -*-
"""
WordCell: الخلية اللغوية الأساسية
A dynamic cell that stores a word and its meaning as executable code.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Set


class WordCell:
    """
    خلية الكلمة - تُخزن كلمة ومعناها وسياقاتها ككود قابل للتنفيذ.
    
    A word cell stores:
    - The word itself
    - Its meanings (as concept cell IDs)
    - Example contexts
    - Usage frequency
    - Learned timestamp
    """
    
    def __init__(self, 
                 word: str, 
                 meanings: List[str] = None,
                 pos: str = None,  # Part of Speech: NOUN, VERB, ADJ, etc.
                 examples: List[str] = None,
                 metadata: Dict = None):
        
        self.word = word
        self.id = f"word_{self._normalize(word)}"
        self.meanings = meanings or []  # Links to concept cells
        self.pos = pos  # Part of Speech
        self.examples = examples or []
        self.metadata = metadata or {}
        
        # Stats
        self.frequency = 0
        self.last_used = None
        self.learned_at = datetime.now().isoformat()
        
        # Connections to other words
        self.synonyms: Set[str] = set()  # Similar words
        self.antonyms: Set[str] = set()  # Opposite words
        self.related: Dict[str, float] = {}  # word_id -> strength
    
    def _normalize(self, word: str) -> str:
        """Normalize word for ID creation."""
        return word.lower().strip().replace(" ", "_")
    
    def use(self):
        """Record usage of this word."""
        self.frequency += 1
        self.last_used = datetime.now().isoformat()
    
    def add_example(self, sentence: str):
        """Add a usage example."""
        if sentence not in self.examples:
            self.examples.append(sentence)
            # Keep only last 10 examples
            if len(self.examples) > 10:
                self.examples = self.examples[-10:]
    
    def add_synonym(self, word_id: str):
        """Add a synonym."""
        self.synonyms.add(word_id)
    
    def add_related(self, word_id: str, strength: float = 0.5):
        """Add a related word with connection strength."""
        self.related[word_id] = strength
    
    def to_source_code(self) -> str:
        """
        Serialize this word cell to executable Python code.
        This is the key feature: words are stored as code, not data.
        """
        meanings_str = json.dumps(self.meanings, ensure_ascii=False)
        examples_str = json.dumps(self.examples, ensure_ascii=False, indent=2)
        synonyms_str = json.dumps(list(self.synonyms), ensure_ascii=False)
        related_str = json.dumps(self.related, ensure_ascii=False)
        metadata_str = json.dumps(self.metadata, ensure_ascii=False)
        
        code = f'''
# --- WORD CELL: {self.word} ---
{self.id} = WordCell(
    word="{self.word}",
    meanings={meanings_str},
    pos="{self.pos or 'UNKNOWN'}",
    examples={examples_str},
    metadata={metadata_str}
)
{self.id}.frequency = {self.frequency}
{self.id}.learned_at = "{self.learned_at}"
{self.id}.synonyms = set({synonyms_str})
{self.id}.related = {related_str}
'''
        return code
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "id": self.id,
            "word": self.word,
            "meanings": self.meanings,
            "pos": self.pos,
            "examples": self.examples,
            "frequency": self.frequency,
            "synonyms": list(self.synonyms),
            "related": self.related,
            "learned_at": self.learned_at
        }
    
    def __repr__(self):
        return f"WordCell('{self.word}', freq={self.frequency}, pos={self.pos})"


# === Quick Test ===
if __name__ == "__main__":
    # Create a word
    w = WordCell(
        word="تفاح", 
        meanings=["concept_fruit"],
        pos="NOUN",
        examples=["أكلت التفاح", "التفاح أحمر"]
    )
    w.use()
    w.use()
    w.add_synonym("word_برتقال")
    w.add_related("word_فاكهة", 0.9)
    
    print(w.to_source_code())
