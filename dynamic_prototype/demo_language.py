import sys
import os
import io

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from dynamic_prototype.vocab import TinyVocab
from dynamic_prototype.dynamic_rnn import DynamicLanguageModel
import random
import time

def softmax(x):
    e_x = [math.exp(i) for i in x]
    sum_e_x = sum(e_x)
    return [i / sum_e_x for i in e_x]

import math

def hack_weights_for_demo(model, vocab):
    """
    Since we can't train this model in 5 seconds provided, we will 'surgically' 
    inject personality into the genes to demonstrate the concept.
    
    We want:
    "System" -> "is" (Always)
    "is" -> 
        if PANIC: "failing", "critical", "danger"
        if CALM:  "working", "stable", "good"
    """
    print("👨‍⚕️ Performing Surgical Weight Injection (Manual Training)...")
    
    # Let's target the Output Weights (W_ho)
    # This is a simplification. In a real RNN, hidden state dynamics matter.
    # Here we assume hidden state largely preserves the input word identity for this rough demo.
    
    sys_idx = vocab.word2idx["system"]
    is_idx = vocab.word2idx["is"]
    
    fail_idx = vocab.word2idx["failing"]
    crit_idx = vocab.word2idx["critical"]
    dang_idx = vocab.word2idx["danger"]
    
    work_idx = vocab.word2idx["working"]
    stab_idx = vocab.word2idx["stable"]
    good_idx = vocab.word2idx["good"]
    
    # 1. Promote "System" -> "is" regardless of context
    # We'll make the weights corresponding to 'is' very high when input was 'system'
    # NOTE: This is a hacky approximation because we don't know exactly which hidden neuron represents "system".
    # Instead, we will hack the GENES themselves to bias towards specific words based on context signal.
    
    # Strategy: We will hack the OUTPUT genes for specific words.
    # The gene for "Failing" will have high SENSITIVITY to Panic (positive signal).
    # The gene for "Working" will have negative SENSITIVITY to Panic (prefers low/negative signal).
    
    # Let's modify the Bias Genes of the Output Layer directly for the target words.
    # W_ho [output_vocab_idx] [hidden_idx]
    
    # Make "Failing" gene highly reactive to positive signal (Panic)
    for h in range(model.hidden_size):
        # "Failing" weights
        gene = model.W_ho[fail_idx][h]
        gene.base_val = -2.0 # Default: Don't say failing
        gene.sensitivity = 5.0 # BUT if Panic (Signal > 0), boost this massively!
        
        # "Critical" weights
        gene = model.W_ho[crit_idx][h]
        gene.base_val = -2.0
        gene.sensitivity = 4.0
        
        # "Working" weights
        gene = model.W_ho[work_idx][h]
        gene.base_val = 1.0 # Default: Working is good
        gene.sensitivity = -5.0 # If Panic (Signal > 0), kill this weight!
        
        # "Stable" weights
        gene = model.W_ho[stab_idx][h]
        gene.base_val = 1.0
        gene.sensitivity = -5.0

    print("✅ Surgery Complete. The model now has a 'Personality'.")

def generate_sentence(model, vocab, start_word, context_val, max_len=5):
    current_word = start_word
    sentence = [current_word]
    
    # Initial hidden state (zeros)
    h_state = [0.0] * model.hidden_size
    
    for _ in range(max_len):
        input_vec = vocab.encode(current_word)
        logits, h_state = model.predict(input_vec, h_state, context_val)
        
        # Simple Greedy Decoding (Pick max)
        probs = softmax(logits)
        best_idx = probs.index(max(probs))
        next_word = vocab.idx_to_word(best_idx)
        
        sentence.append(next_word)
        current_word = next_word
        
        if next_word == "<END>":
            break
            
    return " ".join(sentence)

def demo_language_layer():
    print("🗣️ INITIALIZING DYNAMIC LINGUISTIC LAYER...")
    
    vocab = TinyVocab()
    model = DynamicLanguageModel(vocab.size, hidden_size=8)
    
    # Inject Logic
    hack_weights_for_demo(model, vocab)
    
    print("\n🧪 STARTING GENERATION TEST")
    print(f"Start Word: 'system'")
    print("="*60)
    
    contexts = [
        ("CALM (Safe)", -0.8),
        ("NEUTRAL", 0.0),
        ("WORRIED", 0.4),
        ("PANIC (Danger)", 0.9)
    ]
    
    for name, signal in contexts:
        # Generate text
        output_text = generate_sentence(model, vocab, "system", signal)
        print(f"Context: {name:<15} (Sig: {signal:+.1f}) | Output: {output_text}")
        time.sleep(0.5)
        
    print("="*60)
    print("Notice how the sentence completes differently based on the 'Hormonal' signal.")

if __name__ == "__main__":
    demo_language_layer()
