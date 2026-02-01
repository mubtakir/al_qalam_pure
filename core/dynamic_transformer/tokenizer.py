# -*- coding: utf-8 -*-
"""
Arabic Tokenizer and Dataset
Tokenizer عربي ومُحمّل بيانات للتدريب
"""

import os
import re
import json
from typing import List, Dict, Iterator, Tuple
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader


class ArabicTokenizer:
    """
    Tokenizer بسيط للغة العربية
    - Word-level tokenization
    - Handles Arabic text normalization
    """
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.word2idx: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.word_freq: Counter = Counter()
    
    def normalize(self, text: str) -> str:
        """Normalize Arabic text."""
        # Remove diacritics (tashkeel)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        # Normalize alef
        text = re.sub(r'[إأآا]', 'ا', text)
        # Normalize ya
        text = re.sub(r'ى', 'ي', text)
        # Normalize ta marbuta
        text = re.sub(r'ة', 'ه', text)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = self.normalize(text)
        # Extract Arabic words and punctuation
        tokens = re.findall(r'[\u0600-\u06FF]+|[a-zA-Z]+|[0-9]+|[.,!?;:]', text)
        return tokens
    
    def build_vocab(self, texts: Iterator[str], min_freq: int = 2):
        """Build vocabulary from texts."""
        print("[TOKENIZER] Building vocabulary...")
        
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Add most common words to vocab
        for word, freq in self.word_freq.most_common(self.vocab_size - 4):
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                if idx < self.vocab_size:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
        
        print(f"[TOKENIZER] Vocabulary: {len(self.word2idx)} words")
    
    def encode(self, text: str, max_len: int = None) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 1 = <UNK>
        
        if max_len:
            ids = ids[:max_len]
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.idx2word.get(i, "<UNK>") for i in ids]
        return " ".join(tokens)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[TOKENIZER] Saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        print(f"[TOKENIZER] Loaded {len(self.word2idx)} words from {path}")


class TextDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(self, 
                 file_path: str,
                 tokenizer: ArabicTokenizer,
                 seq_len: int = 128,
                 max_samples: int = None):
        """
        Args:
            file_path: Path to text file
            tokenizer: Tokenizer to use
            seq_len: Sequence length for training
            max_samples: Max samples to load (for testing)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples: List[List[int]] = []
        
        print(f"[DATASET] Loading from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            buffer = []
            for i, line in enumerate(f):
                if max_samples and len(self.samples) >= max_samples:
                    break
                
                # Tokenize line
                tokens = tokenizer.encode(line.strip())
                buffer.extend(tokens)
                
                # Create samples from buffer
                while len(buffer) >= seq_len + 1:
                    self.samples.append(buffer[:seq_len + 1])
                    buffer = buffer[seq_len // 2:]  # Overlap
                
                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i+1} lines, {len(self.samples)} samples")
        
        print(f"[DATASET] Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        input_ids = torch.tensor(sample[:-1], dtype=torch.long)
        labels = torch.tensor(sample[1:], dtype=torch.long)
        return input_ids, labels


def create_dataloader(file_path: str,
                      tokenizer: ArabicTokenizer,
                      batch_size: int = 32,
                      seq_len: int = 128,
                      max_samples: int = None,
                      num_workers: int = 0) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = TextDataset(file_path, tokenizer, seq_len, max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


# === Quick Test ===
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Test tokenizer
    tokenizer = ArabicTokenizer(vocab_size=1000)
    
    # Sample texts
    texts = [
        "القط الأسود يأكل السمك الطازج",
        "الكلب الكبير يلعب في الحديقة",
        "الطفل الصغير يقرأ الكتاب المفيد",
    ]
    
    # Build vocab
    tokenizer.build_vocab(iter(texts), min_freq=1)
    
    # Test encode/decode
    for text in texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        print(f"Original: {text}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")
        print()
