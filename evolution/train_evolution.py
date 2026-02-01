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
    # Shift for stability
    m = max(x)
    e_x = [math.exp(i - m) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

def calculate_loss(logits, target_idx):
    """Negative Log Likelihood."""
    probs = softmax(logits)
    p_target = probs[target_idx]
    # Small epsilon to avoid log(0)
    return -math.log(max(p_target, 1e-9))

def train_evolution():
    print("🧬 STARTING EVOLUTIONARY TRAINING...")
    
    vocab = TinyVocab()
    model = DynamicLanguageModel(vocab.size, hidden_size=8)
    
    # Target phrase we want it to learn
    target_sentence = ["system", "is", "stable"]
    target_indices = [vocab.word2idx[w] for w in target_sentence]
    
    print(f"🎯 Target Phrase: '{' '.join(target_sentence)}'")
    
    optimizer = GeneticOptimizer(
        model, 
        population_size=100, 
        mutation_rate=0.2, 
        mutation_scale=0.2
    )

    generations = 50
    
    for gen in range(generations):
        scores = []
        best_output_text = ""
        best_gen_score = float('inf')
        
        for genome in optimizer.population:
            # Inject Genome
            model.set_genome(genome)
            
            # Evaluate
            total_loss = 0.0
            current_word = "<START>"
            output_tokens = []
            
            # Simple Hidden State
            h_state = [0.0] * model.hidden_size
            
            # Context Signal (Let's say 0.0 for Neutral, to keep it simple for now)
            # In a real scenario, we'd train it to say "stable" only when signal is "Calm".
            context_signal = -0.5 # Calm signal
            
            for target_idx in target_indices:
                input_vec = vocab.encode(current_word)
                logits, h_state = model.predict(input_vec, h_state, context_signal)
                
                loss = calculate_loss(logits, target_idx)
                total_loss += loss
                
                # For visualization, pick the best word
                probs = softmax(logits)
                best_idx = probs.index(max(probs))
                output_tokens.append(vocab.idx_to_word(best_idx))
                
                # Teacher Forcing: Feed the *target* word as next input, purely for training stability
                # Or Autoregression: Feed the *predicted* word?
                # Evolution works better with Autoregression usually, but for short phrase, Teacher Forcing is okay.
                # Let's use Teacher Forcing for "next input" to focus on learning P(next|curr).
                current_word = vocab.idx_to_word(target_indices[target_indices.index(target_idx)]) 
            
            scores.append(total_loss)
            
            if total_loss < best_gen_score:
                best_gen_score = total_loss
                best_output_text = " ".join(output_tokens)
                
        # Evolve
        best_score = optimizer.evolve_step(scores)
        
        if gen % 5 == 0:
            print(f"Gen {gen:03d} | Best Loss: {best_score:.4f} | Output: {best_output_text}")
            
        if best_score < 0.1:
            print(f"Gen {gen:03d} | Converged! Output: {best_output_text}")
            break

    print("="*60)
    print("✅ EVOLUTION COMPLETE.")

    # Final Verification with different signals?
    # For this basic test, we just wanted to prove it can learn AT ALL.
    
if __name__ == "__main__":
    train_evolution()
