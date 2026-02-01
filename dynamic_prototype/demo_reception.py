import sys
import os
import io

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from dynamic_prototype.nano_layer import NanoLayer
import random
import time

def demo_dynamic_weights():
    print("🧬 INITIALIZING DYNAMIC NANO-MODEL...")
    
    # Create a tiny layer: 3 Inputs -> 2 Outputs
    layer = NanoLayer(input_size=3, output_size=2)
    
    # Define a static input (e.g., embedding for word "Apple")
    # In a static model, this input should ALWAYS produce the same output.
    static_input = [1.0, 0.5, -0.5] 
    
    contexts = [
        ("Neutral", 0.0),
        ("Panic", 0.8),
        ("Calm", -0.5),
        ("Excitement", 0.9)
    ]
    
    print(f"\n📦 Fixed Input Vector: {static_input}")
    print("="*60)
    print(f"{'Context':<15} | {'Signal':<8} | {'Output Node 1':<15} | {'Output Node 2':<15} | {'Weight Behavior'}")
    print("-" * 60)

    for name, signal in contexts:
        # Run forward pass with specific context
        output = layer.forward(static_input, signal)
        
        # Check how one specific weight reacted (e.g., Row 0, Col 0)
        sample_weight_gene = layer.weights[0][0]
        curr_w_val = sample_weight_gene.history[-1]
        
        print(f"{name:<15} | {signal:<8.1f} | {output[0]:<15.4f} | {output[1]:<15.4f} | W[0,0]={curr_w_val:.3f}")
        time.sleep(0.5)

    print("="*60)
    print("\n✅ DEMO COMPLETE.")
    print("Notice how the 'Fixed Input' produced DIFFERENT outputs solely because the weights shifted themselves.")
    print("In a standard model, Output would be identical for all rows.")

if __name__ == "__main__":
    demo_dynamic_weights()
