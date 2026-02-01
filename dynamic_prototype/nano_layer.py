from typing import List
from dynamic_prototype.gene import WeightGene

class NanoLayer:
    """
    A layer of neurons where every connection is a WeightGene.
    """
    def __init__(self, input_size: int, output_size: int):
        self.weights: List[List[WeightGene]] = []
        
        # Initialize with random 'genes'
        import random
        for _ in range(output_size):
            row = []
            for _ in range(input_size):
                # Random personality for each synapse
                base = random.uniform(-1.0, 1.0)
                sens = random.uniform(-0.5, 0.5) # Some are reactive, some are stubborn
                row.append(WeightGene(base, sens))
            self.weights.append(row)

    def forward(self, inputs: List[float], context_signal: float) -> List[float]:
        """
        Forward pass.
        CRITICAL DIFFERENCE: We pass 'context_signal' which modifies the weights *before* multiplication.
        """
        outputs = []
        
        for row in self.weights:
            neuron_sum = 0.0
            for i, weight_gene in enumerate(row):
                # 1. Ask the gene: "What is your value right now given this context?"
                w_val = weight_gene.compute(context_signal)
                
                # 2. Standard weighted sum
                neuron_sum += w_val * inputs[i]
            
            outputs.append(neuron_sum)
            
        return outputs
