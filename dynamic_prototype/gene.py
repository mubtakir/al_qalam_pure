import math
import random

class WeightGene:
    """
    A 'Living Weight'. 
    Instead of being a static float, this is an object that computes its value
    based on the current context signal.
    """
    def __init__(self, base_val: float, sensitivity: float, bias: float = 0.0):
        self.base_val = base_val        # The "resting" state of the weight
        self.sensitivity = sensitivity  # How much it reacts to context (Elasticity)
        self.bias = bias
        self.history = []               # For visualization/debugging

    def compute(self, context_signal: float) -> float:
        """
        Calculates the effective weight at this specific moment.
        Formula: W_eff = tanh(Base + (Sensitivity * Context) + Bias)
        """
        # The weight 'reacts' to the context signal
        raw_val = self.base_val + (self.sensitivity * context_signal) + self.bias
        
        # Non-linearity to keep it stable (like in biological synapses)
        effective_weight = math.tanh(raw_val)
        
        self.history.append(effective_weight)
        return effective_weight

    def mutate(self):
        """Evolutionary Step: Randomly adjust properties."""
        self.base_val += random.uniform(-0.1, 0.1)
        self.sensitivity += random.uniform(-0.05, 0.05)

    def get_params(self):
        return (self.base_val, self.sensitivity, self.bias)

    def set_params(self, params):
        self.base_val, self.sensitivity, self.bias = params
