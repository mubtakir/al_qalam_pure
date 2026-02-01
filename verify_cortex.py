import os
import sys
import io

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.self_writing_model import SelfWritingModel
from core.cortex import Cortex

def verify_cortex():
    print("--- Verification: Al-Qalam Cortex (Thinking Engine) ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    cortex = Cortex(model)
    
    # 1. Trigger thinking
    print("[1] Running Cortex thinking process...")
    thoughts = cortex.think()
    
    if thoughts:
        print("\n[Cortex Inner Thoughts]:")
        for t in thoughts:
            print(f" -> {t}")
        
        # Check for analogy output
        if any("قياس منطقي" in t for t in thoughts):
            print("\n[Success] Cortex discovered cross-domain analogies!")
    else:
        print("\n[Note] No specialized thoughts yet. Ensure concepts were loaded.")

    print("\n--- Cortex Verification Complete ---")

if __name__ == "__main__":
    verify_cortex()
