import math
from typing import List
from core.liquid_engine.gene import WeightGene

class DynamicRNNCell:
    """
    A Recurrent Neural Network cell built with Living Weights.
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.W_ih = [[self._create_gene() for _ in range(input_size)] for _ in range(hidden_size)]
        self.W_hh = [[self._create_gene() for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.bias = [self._create_gene() for _ in range(hidden_size)]
        
    def _create_gene(self):
        import random
        return WeightGene(random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3))

    def forward_step(self, x_t: List[float], h_prev: List[float], context_signal: float) -> List[float]:
        h_next = []
        for i in range(self.hidden_size):
            neuron_act = 0.0
            for j, w_gene in enumerate(self.W_ih[i]):
                neuron_act += w_gene.compute(context_signal) * x_t[j]
            for j, w_gene in enumerate(self.W_hh[i]):
                neuron_act += w_gene.compute(context_signal) * h_prev[j]
            neuron_act += self.bias[i].compute(context_signal)
            h_next.append(math.tanh(neuron_act))
        return h_next

class DynamicLanguageModel:
    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn = DynamicRNNCell(vocab_size, hidden_size)
        self.W_ho = [[self.rnn._create_gene() for _ in range(hidden_size)] for _ in range(vocab_size)]
        # Add Output Bias Genes for reliable firing
        self.out_bias = [self.rnn._create_gene() for _ in range(vocab_size)]

    def predict(self, input_vec: List[float], h_prev: List[float], context_signal: float) -> (List[float], List[float]):
        h_next = self.rnn.forward_step(input_vec, h_prev, context_signal)
        logits = []
        for i in range(self.vocab_size):
            sum_val = 0.0
            for j, w_gene in enumerate(self.W_ho[i]):
                sum_val += w_gene.compute(context_signal) * h_next[j]
            # Add Bias
            sum_val += self.out_bias[i].compute(context_signal)
            logits.append(sum_val)
        return logits, h_next

    def get_genome(self) -> List[tuple]:
        genome = []
        for row in self.rnn.W_ih:
            for gene in row: genome.append(gene.get_params())
        for row in self.rnn.W_hh:
            for gene in row: genome.append(gene.get_params())
        for gene in self.rnn.bias: genome.append(gene.get_params())
        for row in self.W_ho:
            for gene in row: genome.append(gene.get_params())
        return genome

    def set_genome(self, genome: List[tuple]):
        idx = 0
        for row in self.rnn.W_ih:
            for gene in row: gene.set_params(genome[idx]); idx += 1
        for row in self.rnn.W_hh:
            for gene in row: gene.set_params(genome[idx]); idx += 1
        for gene in self.rnn.bias: gene.set_params(genome[idx]); idx += 1
        for row in self.W_ho:
            for gene in row: gene.set_params(genome[idx]); idx += 1
