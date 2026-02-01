import sys
import os
import io
import math

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from dynamic_prototype.vocab import TinyVocab
from dynamic_prototype.dynamic_rnn import DynamicLanguageModel
from evolution.optimizer import GeneticOptimizer

def softmax(x):
    m = max(x)
    e_x = [math.exp(i - m) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

def calculate_loss(logits, target_idx):
    probs = softmax(logits)
    p_target = probs[target_idx]
    return -math.log(max(p_target, 1e-9))

def train_evolution_dynamic():
    print("🧬 STARTING DYNAMIC EVOLUTION STRESS TEST...")
    print("Goal: Learn to say 'Stable' in Calm context AND 'Failing' in Panic context.")
    
    vocab = TinyVocab()
    model = DynamicLanguageModel(vocab.size, hidden_size=8)
    
    # Two conflicting targets based on context
    scenarios = [
        {"context": -0.8, "target": ["system", "is", "stable"]}, # Calm
        {"context": +0.9, "target": ["system", "is", "failing"]} # Panic
    ]
    
    optimizer = GeneticOptimizer(
        model, 
        population_size=150, 
        mutation_rate=0.3, # Higher mutation for complex logic
        mutation_scale=0.3
    )

    generations = 100
    
    for gen in range(generations):
        scores = []
        best_output_str = ""
        best_gen_score = float('inf')
        
        for genome in optimizer.population:
            model.set_genome(genome)
            total_loss = 0.0
            outputs = []
            
            # Evaluate on BOTH scenarios
            for scen in scenarios:
                target_indices = [vocab.word2idx[w] for w in scen["target"]]
                ctx = scen["context"]
                
                output_tokens = []
                current_word = "<START>"
                h_state = [0.0] * model.hidden_size
                
                for target_idx in target_indices:
                    input_vec = vocab.encode(current_word)
                    logits, h_state = model.predict(input_vec, h_state, ctx)
                    total_loss += calculate_loss(logits, target_idx)
                    
                    probs = softmax(logits)
                    best_idx = probs.index(max(probs))
                    output_tokens.append(vocab.idx_to_word(best_idx))
                    
                    # Teacher Forcing
                    current_word = vocab.idx_to_word(target_indices[target_indices.index(target_idx)]) 
                
                outputs.append(" ".join(output_tokens))
            
            scores.append(total_loss)
            
            if total_loss < best_gen_score:
                best_gen_score = total_loss
                best_output_str = f"Calm->[{outputs[0]}] | Panic->[{outputs[1]}]"
                
        # Evolve
        best_score = optimizer.evolve_step(scores)
        
        if gen % 10 == 0:
            print(f"Gen {gen:03d} | Loss: {best_score:.4f} | {best_output_str}")
            
        if best_score < 0.2:
            print(f"Gen {gen:03d} | Converged! {best_output_str}")
            break

    print("="*60)
    print("✅ COMPLEX EVOLUTION COMPLETE.")

    
if __name__ == "__main__":
    train_evolution_dynamic()
