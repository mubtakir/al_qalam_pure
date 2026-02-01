# -*- coding: utf-8 -*-
"""
PatternCell: خلية النمط اللغوي
Stores sentence patterns that can be used to generate new sentences.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class PatternCell:
    """
    خلية النمط - تُخزن نمط جملة يمكن استخدامه للتوليد.
    
    Example Pattern: ["NOUN", "VERB", "NOUN"]
    Matches: "القط يأكل السمك", "الطفل يقرأ الكتاب"
    """
    
    def __init__(self,
                 pattern_id: str,
                 structure: List[str],  # ["NOUN", "VERB", "NOUN"]
                 examples: List[str] = None,
                 template: str = None,  # "{0} {1} {2}"
                 metadata: Dict = None):
        
        self.id = f"pattern_{pattern_id}"
        self.structure = structure
        self.examples = examples or []
        self.template = template or " ".join([f"{{{i}}}" for i in range(len(structure))])
        self.metadata = metadata or {}
        
        # Stats
        self.frequency = 0
        self.success_rate = 1.0  # How often generated sentences are good
        self.learned_at = datetime.now().isoformat()
        
        # Slots: which words commonly fill each position
        # slots[position_idx] = {word_id: frequency}
        self.slots: Dict[int, Dict[str, int]] = {i: {} for i in range(len(structure))}
    
    def record_example(self, words: List[str]):
        """Record which words filled each slot in an example."""
        for i, word in enumerate(words):
            if i < len(self.slots):
                word_id = f"word_{word}"
                self.slots[i][word_id] = self.slots[i].get(word_id, 0) + 1
    
    def get_best_candidates(self, slot_idx: int, n: int = 5) -> List[str]:
        """Get the most common words for a slot."""
        if slot_idx >= len(self.slots):
            return []
        slot = self.slots[slot_idx]
        sorted_words = sorted(slot.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:n]]
    
    def matches(self, pos_tags: List[str]) -> bool:
        """Check if a list of POS tags matches this pattern."""
        if len(pos_tags) != len(self.structure):
            return False
        return all(p == s or s == "ANY" for p, s in zip(pos_tags, self.structure))
    
    def generate(self, words: List[str]) -> str:
        """Generate a sentence from the pattern and words."""
        if len(words) != len(self.structure):
            raise ValueError(f"Expected {len(self.structure)} words, got {len(words)}")
        return self.template.format(*words)
    
    def use(self, success: bool = True):
        """Record usage of this pattern."""
        self.frequency += 1
        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
    
    def to_source_code(self) -> str:
        """Serialize to executable Python code."""
        structure_str = json.dumps(self.structure, ensure_ascii=False)
        examples_str = json.dumps(self.examples, ensure_ascii=False)
        slots_str = json.dumps(self.slots, ensure_ascii=False)
        metadata_str = json.dumps(self.metadata, ensure_ascii=False)
        
        code = f'''
# --- PATTERN CELL: {self.id} ---
{self.id} = PatternCell(
    pattern_id="{self.id.replace('pattern_', '')}",
    structure={structure_str},
    examples={examples_str},
    template="{self.template}",
    metadata={metadata_str}
)
{self.id}.frequency = {self.frequency}
{self.id}.success_rate = {self.success_rate}
{self.id}.learned_at = "{self.learned_at}"
{self.id}.slots = {slots_str}
'''
        return code
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "id": self.id,
            "structure": self.structure,
            "template": self.template,
            "examples": self.examples,
            "frequency": self.frequency,
            "success_rate": self.success_rate,
            "slots": self.slots
        }
    
    def __repr__(self):
        struct = "->".join(self.structure)
        return f"PatternCell('{struct}', freq={self.frequency})"


# === Quick Test ===
if __name__ == "__main__":
    # Create a pattern: NOUN VERB NOUN
    p = PatternCell(
        pattern_id="nvn_1",
        structure=["NOUN", "VERB", "NOUN"],
        examples=["القط يأكل السمك", "الطفل يقرأ الكتاب"]
    )
    
    # Record slot usage
    p.record_example(["القط", "يأكل", "السمك"])
    p.record_example(["القط", "يشرب", "الماء"])
    p.record_example(["الطفل", "يقرأ", "الكتاب"])
    
    print("Pattern:", p)
    print("Best candidates for slot 0:", p.get_best_candidates(0))
    print("Generated:", p.generate(["الكلب", "يحب", "اللعب"]))
    print()
    print(p.to_source_code())
