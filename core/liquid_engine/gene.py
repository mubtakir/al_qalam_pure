import math
import random

class WeightGene:
    """
    A 'Living Weight' for the Liquid Engine.
    Computes its value based on context signal: W = tanh(Base + Sens*Signal + Bias)
    """
    def __init__(self, base_val: float, sensitivity: float, bias: float = 0.0):
        self.base_val = base_val
        self.sensitivity = sensitivity 
        self.bias = bias
        self.history = []

    def compute(self, context_signal: float) -> float:
        raw_val = self.base_val + (self.sensitivity * context_signal) + self.bias
        effective_weight = math.tanh(raw_val)
        self.history.append(effective_weight)
        return effective_weight

    def mutate(self):
        self.base_val += random.uniform(-0.1, 0.1)
        self.sensitivity += random.uniform(-0.05, 0.05)

    def get_params(self):
        return (self.base_val, self.sensitivity, self.bias)

    def set_params(self, params):
        self.base_val, self.sensitivity, self.bias = params
