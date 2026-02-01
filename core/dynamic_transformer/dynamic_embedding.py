# -*- coding: utf-8 -*-
"""
DynamicEmbedding: Embedding ديناميكي
An embedding layer that can learn new words without retraining.

Key Innovation: Words are added dynamically, embeddings stored as code.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .weight_cell import WeightCell


class DynamicEmbedding:
    """
    Embedding ديناميكي - يتعلم كلمات جديدة بدون إعادة تدريب.
    
    Unlike traditional embeddings:
    - Can add new words at runtime
    - Embeddings stored as Python code
    - No vocab_size limit
    """
    
    def __init__(self, 
                 dim: int = 256,
                 name: str = "embedding",
                 persist_path: str = None):
        """
        Initialize dynamic embedding.
        
        Args:
            dim: Embedding dimension
            name: Name for identification
            persist_path: Path to save embeddings.py
        """
        self.dim = dim
        self.name = name
        self.persist_path = persist_path
        
        # Word to embedding mapping
        self.embeddings: Dict[str, WeightCell] = {}
        
        # Word to index (for position)
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        
        # Special tokens
        self._add_special_tokens()
        
        # Statistics
        self.created_at = datetime.now().isoformat()
    
    def _add_special_tokens(self):
        """Add special tokens."""
        special = ["<PAD>", "<UNK>", "<START>", "<END>", "<MASK>"]
        for token in special:
            self.add_word(token)
    
    def add_word(self, word: str, embedding: np.ndarray = None) -> WeightCell:
        """
        Add a new word to the vocabulary.
        
        Args:
            word: The word to add
            embedding: Optional pre-computed embedding
        """
        if word in self.embeddings:
            return self.embeddings[word]
        
        # Create embedding
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx] = word
        
        if embedding is not None:
            cell = WeightCell(
                shape=(self.dim,),
                name=f"emb_{word[:20]}",
                values=embedding
            )
        else:
            cell = WeightCell(
                shape=(self.dim,),
                name=f"emb_{word[:20]}",
                init_method="normal"
            )
        
        self.embeddings[word] = cell
        return cell
    
    def get_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a word."""
        if word in self.embeddings:
            return self.embeddings[word].values
        else:
            # Return <UNK> embedding
            return self.embeddings["<UNK>"].values
    
    def embed_sentence(self, words: List[str]) -> np.ndarray:
        """
        Embed a sentence (list of words).
        
        Returns: (seq_len, dim) array
        """
        embeddings = []
        for word in words:
            embeddings.append(self.get_embedding(word))
        return np.array(embeddings)
    
    def learn_from_context(self, 
                           target_word: str, 
                           context_words: List[str],
                           learning_rate: float = 0.01):
        """
        Learn embedding from context (simplified Word2Vec-like).
        
        Args:
            target_word: The word to learn
            context_words: Surrounding words
        """
        # Ensure words exist
        if target_word not in self.embeddings:
            self.add_word(target_word)
        
        for ctx in context_words:
            if ctx not in self.embeddings:
                self.add_word(ctx)
        
        # Average context embeddings
        context_emb = np.mean([self.get_embedding(w) for w in context_words], axis=0)
        
        # Move target closer to context
        target_cell = self.embeddings[target_word]
        delta = target_cell.values - context_emb
        target_cell.update(delta, learning_rate)
    
    def similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words."""
        if word1 not in self.embeddings or word2 not in self.embeddings:
            return 0.0
        
        v1 = self.get_embedding(word1)
        v2 = self.get_embedding(word2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def most_similar(self, word: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        if word not in self.embeddings:
            return []
        
        similarities = []
        for other_word in self.embeddings:
            if other_word != word:
                sim = self.similarity(word, other_word)
                similarities.append((other_word, sim))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]
    
    def to_source_code(self) -> str:
        """Generate embeddings as executable Python code."""
        header = f'''# -*- coding: utf-8 -*-
"""
Al-Qalam Dynamic Embeddings
Auto-generated - DO NOT EDIT MANUALLY
Generated: {datetime.now().isoformat()}
Vocabulary: {len(self.embeddings)} words
Dimension: {self.dim}
"""

import numpy as np
from core.dynamic_transformer.weight_cell import WeightCell

# === EMBEDDINGS ===
'''
        code = header
        
        # Sort by frequency/importance (index order)
        for word, cell in self.embeddings.items():
            # Escape special characters in word
            safe_word = word.replace('"', '\\"')
            values_str = np.array2string(
                cell.values, separator=',', 
                threshold=1000, max_line_width=200
            )
            code += f'''
# Word: {safe_word}
embeddings["{safe_word}"] = np.array({values_str})
'''
        
        # Add index
        code += "\n# === WORD INDEX ===\n"
        code += "WORD2IDX = {\n"
        for word, idx in self.word2idx.items():
            safe_word = word.replace('"', '\\"')
            code += f'    "{safe_word}": {idx},\n'
        code += "}\n"
        
        return code
    
    def persist(self) -> bool:
        """Save embeddings to file."""
        if not self.persist_path:
            return False
        
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        
        code = self.to_source_code()
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return True
    
    def __len__(self):
        return len(self.embeddings)
    
    def __contains__(self, word):
        return word in self.embeddings
    
    def __repr__(self):
        return f"DynamicEmbedding(vocab={len(self.embeddings)}, dim={self.dim})"


# === Quick Test ===
if __name__ == "__main__":
    emb = DynamicEmbedding(dim=64)
    
    # Add words
    emb.add_word("القط")
    emb.add_word("الكلب")
    emb.add_word("الحيوان")
    emb.add_word("يأكل")
    emb.add_word("السمك")
    
    print(emb)
    print()
    
    # Learn from context
    emb.learn_from_context("القط", ["الحيوان", "يأكل", "السمك"])
    emb.learn_from_context("الكلب", ["الحيوان", "يأكل"])
    
    # Similarity
    print("Similarity(القط, الكلب):", emb.similarity("القط", "الكلب"))
    print("Similarity(القط, يأكل):", emb.similarity("القط", "يأكل"))
    print()
    
    # Embed sentence
    sentence = ["القط", "يأكل", "السمك"]
    embedded = emb.embed_sentence(sentence)
    print("Sentence:", sentence)
    print("Embedded shape:", embedded.shape)
