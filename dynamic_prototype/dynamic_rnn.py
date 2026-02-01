import math
from typing import List
from dynamic_prototype.gene import WeightGene

class DynamicRNNCell:
    """
    A Recurrent Neural Network cell built with Living Weights.
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + bias)
    """
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Input -> Hidden Weights
        self.W_ih = [[self._create_gene() for _ in range(input_size)] for _ in range(hidden_size)]
        
        # Hidden -> Hidden Weights
        self.W_hh = [[self._create_gene() for _ in range(hidden_size)] for _ in range(hidden_size)]
        
        # Bias
        self.bias = [self._create_gene() for _ in range(hidden_size)]
        
        # Output Layer (Hidden -> Vocab) - we include it here for simplicity in this demo
        # In a real model, this would be a separate layer
        
    def _create_gene(self):
        import random
        # Initialize with slight randomness
        return WeightGene(random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3))

    def forward_step(self, x_t: List[float], h_prev: List[float], context_signal: float) -> List[float]:
        """Calculates the next hidden state."""
        h_next = []
        
        for i in range(self.hidden_size):
            neuron_act = 0.0
            
            # W_ih * x_t
            for j, w_gene in enumerate(self.W_ih[i]):
                neuron_act += w_gene.compute(context_signal) * x_t[j]
            
            # W_hh * h_prev
            for j, w_gene in enumerate(self.W_hh[i]):
                neuron_act += w_gene.compute(context_signal) * h_prev[j]
                
            # Bias
            neuron_act += self.bias[i].compute(context_signal)
            
            # Activation
            h_next.append(math.tanh(neuron_act))
            
        return h_next

class DynamicLanguageModel:
    def __init__(self, vocab_size: int, hidden_size: int):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn = DynamicRNNCell(vocab_size, hidden_size)
        
        # Hidden -> Output Weights
        self.W_ho = [[self.rnn._create_gene() for _ in range(hidden_size)] for _ in range(vocab_size)]

    def predict(self, input_vec: List[float], h_prev: List[float], context_signal: float) -> (List[float], List[float]):
        """Runs one step: Returns (output_logits, next_hidden_state)."""
        
        # 1. RNN Step
        h_next = self.rnn.forward_step(input_vec, h_prev, context_signal)
        
        # 2. Output Projection
        logits = []
        for i in range(self.vocab_size):
            sum_val = 0.0
            for j, w_gene in enumerate(self.W_ho[i]):
                sum_val += w_gene.compute(context_signal) * h_next[j]
            logits.append(sum_val)
            
        return logits, h_next
    
    def force_set_weights_for_demo(self, vocab):
        """
        Manually guiding the weights.
        """
        pass 

    def get_genome(self) -> List[tuple]:
        """Extracts all genes from the model as a flat list."""
        genome = []
        # RNN Genes
        for row in self.rnn.W_ih:
            for gene in row: genome.append(gene.get_params())
        for row in self.rnn.W_hh:
            for gene in row: genome.append(gene.get_params())
        for gene in self.rnn.bias: genome.append(gene.get_params())
        
        # Output Genes
        for row in self.W_ho:
            for gene in row: genome.append(gene.get_params())
            
        return genome

    def set_genome(self, genome: List[tuple]):
        """Injects a genome into the model."""
        idx = 0
        # RNN Genes
        for row in self.rnn.W_ih:
            for gene in row: 
                gene.set_params(genome[idx])
                idx += 1
        for row in self.rnn.W_hh:
            for gene in row: 
                gene.set_params(genome[idx])
                idx += 1
        for gene in self.rnn.bias: 
            gene.set_params(genome[idx])
            idx += 1
            
        # Output Genes
        for row in self.W_ho:
            for gene in row: 
                gene.set_params(genome[idx])
                idx += 1
