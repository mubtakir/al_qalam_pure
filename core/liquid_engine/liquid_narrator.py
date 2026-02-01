import math
from typing import List
from core.liquid_engine.vocab import TinyVocab
from core.liquid_engine.dynamic_rnn import DynamicLanguageModel

class LiquidNarrator:
    """
    The Voice of Al-Qalam that changes tone based on internal system mood.
    Wraps the Dynamic Codes-Weights Model.
    """
    def __init__(self):
        self.vocab = TinyVocab()
        self.model = DynamicLanguageModel(self.vocab.size, hidden_size=8)
        self.current_mood = 0.0 # -1.0 (Calm) to +1.0 (Panic)
        
        # Load "Pre-Evolved" Knowledge 
        # (We use the logic derived from evolution: Sensitive genes for stable/failing)
        self._inject_evolved_instincts()

    def set_mood(self, mood: float):
        """Sets the hormonal state of the brain (-1.0 to 1.0)."""
        self.current_mood = max(-1.0, min(1.0, mood))

    def _inject_evolved_instincts(self):
        """
        Manually applies the 'lessons' learned from the evolutionary training.
        Calm (-0.8) -> Stable
        Panic (+0.9) -> Failing
        We use Output BIAS genes now for reliability.
        """
        fail_idx = self.vocab.word2idx.get("failing")
        stab_idx = self.vocab.word2idx.get("stable")
        
        # Apply the "Evolved Sensitivity"
        # Stable Gene: High activation when signal is LOW (Negative)
        if stab_idx:
            gene = self.model.out_bias[stab_idx]
            gene.base_val = 2.0 # Strong base preference
            gene.sensitivity = -4.0 # inhibited by Panic, boosted by Calm
            
        # Target: Failing
        if fail_idx:
            gene = self.model.out_bias[fail_idx]
            gene.base_val = -2.0 # Default: Don't say it
            gene.sensitivity = 5.0 # Positive sensitivity (Likes Panic)

    def speak(self, start_word="system", max_len=6) -> str:
        """Generates a sentence starting with start_word, colored by current mood."""
        current_word = start_word
        sentence = [current_word]
        h_state = [0.0] * self.model.hidden_size
        
        for _ in range(max_len):
            input_vec = self.vocab.encode(current_word)
            logits, h_state = self.model.predict(input_vec, h_state, self.current_mood)
            
            # Softmax & Argmax
            exps = [math.exp(x) for x in logits]
            sum_exps = sum(exps)
            probs = [x/sum_exps for x in exps]
            
            best_idx = probs.index(max(probs))
            next_word = self.vocab.idx_to_word(best_idx)
            
            if next_word == "<END>":
                break
            
            sentence.append(next_word)
            current_word = next_word
            
        return " ".join(sentence)
