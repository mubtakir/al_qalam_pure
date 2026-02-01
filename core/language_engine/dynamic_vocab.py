# -*- coding: utf-8 -*-
"""
DynamicVocab: مدير المفردات الديناميكي
Manages a vocabulary that grows and persists itself as code.
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from .word_cell import WordCell


class DynamicVocab:
    """
    مدير المفردات الديناميكي - يتعلم كلمات جديدة ويحفظها ككود.
    
    Features:
    - Learn new words from text
    - Persist vocabulary as Python code
    - Find words by various criteria
    - Self-writing: vocab grows by writing more code
    """
    
    def __init__(self, persist_path: str = None):
        """
        Initialize the vocabulary manager.
        
        Args:
            persist_path: Path to save vocab.py (e.g., vault/language/vocab.py)
        """
        self.words: Dict[str, WordCell] = {}  # word_id -> WordCell
        self.word_index: Dict[str, str] = {}  # actual_word -> word_id
        self.persist_path = persist_path
        
        # POS tagging (simple rule-based for Arabic)
        self._pos_rules = {
            "VERB": ["يـ", "تـ", "نـ", "أ"],  # Prefixes
            "ADJ": ["ـي", "ـة", "ـاء"],  # Suffixes
            "NOUN": []  # Default
        }
    
    def add_word(self, 
                 word: str, 
                 meanings: List[str] = None,
                 pos: str = None,
                 examples: List[str] = None) -> WordCell:
        """Add a new word to vocabulary."""
        
        # Check if already exists
        if word in self.word_index:
            cell = self.words[self.word_index[word]]
            # Update with new info
            if meanings:
                cell.meanings.extend([m for m in meanings if m not in cell.meanings])
            if examples:
                for ex in examples:
                    cell.add_example(ex)
            return cell
        
        # Create new word cell
        cell = WordCell(
            word=word,
            meanings=meanings,
            pos=pos or self._guess_pos(word),
            examples=examples
        )
        
        self.words[cell.id] = cell
        self.word_index[word] = cell.id
        
        return cell
    
    def _guess_pos(self, word: str) -> str:
        """Guess part of speech using simple rules."""
        # Check for verb prefixes
        for prefix in self._pos_rules["VERB"]:
            if word.startswith(prefix.replace("ـ", "")):
                return "VERB"
        
        # Check for adjective suffixes
        for suffix in self._pos_rules["ADJ"]:
            if word.endswith(suffix.replace("ـ", "")):
                return "ADJ"
        
        return "NOUN"  # Default
    
    def get_word(self, word: str) -> Optional[WordCell]:
        """Get a word cell by the actual word."""
        word_id = self.word_index.get(word)
        return self.words.get(word_id) if word_id else None
    
    def get_by_id(self, word_id: str) -> Optional[WordCell]:
        """Get a word cell by ID."""
        return self.words.get(word_id)
    
    def find_by_pos(self, pos: str) -> List[WordCell]:
        """Find all words with a specific POS."""
        return [w for w in self.words.values() if w.pos == pos]
    
    def find_by_meaning(self, meaning_id: str) -> List[WordCell]:
        """Find all words with a specific meaning."""
        return [w for w in self.words.values() if meaning_id in w.meanings]
    
    def learn_from_text(self, text: str, pos_tags: List[Tuple[str, str]] = None):
        """
        Learn words from a text.
        
        Args:
            text: The text to learn from
            pos_tags: Optional list of (word, pos) tuples
        """
        # Simple tokenization for Arabic
        words = re.findall(r'[\u0600-\u06FF]+', text)
        
        if pos_tags:
            # Use provided POS tags
            for word, pos in pos_tags:
                self.add_word(word, pos=pos, examples=[text])
        else:
            # Learn with guessed POS
            for word in words:
                self.add_word(word, examples=[text])
    
    def use_word(self, word: str):
        """Record usage of a word."""
        cell = self.get_word(word)
        if cell:
            cell.use()
    
    def get_most_frequent(self, n: int = 10) -> List[WordCell]:
        """Get the most frequently used words."""
        sorted_words = sorted(
            self.words.values(), 
            key=lambda w: w.frequency, 
            reverse=True
        )
        return sorted_words[:n]
    
    def to_source_code(self) -> str:
        """Generate the full vocabulary as executable Python code."""
        header = '''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Vocabulary
Auto-generated - DO NOT EDIT MANUALLY
Generated: {timestamp}
Total Words: {count}
"""

from core.language_engine.word_cell import WordCell

# === VOCABULARY ===
'''.format(timestamp=datetime.now().isoformat(), count=len(self.words))
        
        # Sort by frequency (most used first)
        sorted_words = sorted(
            self.words.values(),
            key=lambda w: w.frequency,
            reverse=True
        )
        
        code = header
        for word in sorted_words:
            code += word.to_source_code()
            code += "\n"
        
        # Add index
        code += "\n# === WORD INDEX ===\n"
        code += "WORD_INDEX = {\n"
        for word, word_id in self.word_index.items():
            code += f'    "{word}": "{word_id}",\n'
        code += "}\n"
        
        return code
    
    def persist(self) -> bool:
        """Save vocabulary to file."""
        if not self.persist_path:
            return False
        
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        code = self.to_source_code()
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return True
    
    def __len__(self):
        return len(self.words)
    
    def __repr__(self):
        return f"DynamicVocab({len(self.words)} words)"


# === Quick Test ===
if __name__ == "__main__":
    vocab = DynamicVocab()
    
    # Learn from text
    vocab.learn_from_text("القط الأسود يأكل السمك الطازج")
    vocab.learn_from_text("الكلب الكبير يحب اللعب")
    
    # Use some words
    vocab.use_word("القط")
    vocab.use_word("القط")
    vocab.use_word("يأكل")
    
    print(vocab)
    print("Most frequent:", vocab.get_most_frequent(3))
    print()
    print(vocab.to_source_code())
