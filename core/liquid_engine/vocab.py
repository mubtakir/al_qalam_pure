from typing import List

class TinyVocab:
    def __init__(self):
        # Extended vocab for system integration
        self.words = [
            "<PAD>", "<START>", "<END>", 
            "system", "is", "working", "failing", "stable", "critical",
            "hello", "user", "danger", "safe", "alert", "normal",
            "the", "status", "report", "error", "good",
            "cortex", "active", "thinking", "dreaming", "concept"
        ]
        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.idx2word = {i: w for i, w in enumerate(self.words)}
        self.size = len(self.words)

    def encode(self, word: str) -> List[float]:
        vec = [0.0] * self.size
        idx = self.word2idx.get(word, 0)
        vec[idx] = 1.0
        return vec

    def decode(self, vec: List[float]) -> str:
        idx = vec.index(max(vec))
        return self.idx2word[idx]
    
    def idx_to_word(self, idx: int) -> str:
        return self.idx2word.get(idx, "<UNK>")
