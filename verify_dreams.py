import os
import sys
import io
import time

# Ensure UTF-8 output for Arabic on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.cortex import Cortex

def verify_dreaming():
    print("--- Verification: Al-Qalam Dreaming System ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    cortex = Cortex(model)
    
    print("[1] Simulation: Sleeping and Dreaming for 3 cycles...")
    for i in range(3):
        cortex.think()
        dream = cortex.dreamer.current_dream
        print(f"Cycle {i+1} Dream: {dream}")
        
    print("\n[2] Checking for Strong Hypotheses (Intuitions)...")
    # Force a lot of dreams to ensure one triggers stability > 0.8
    triggered = False
    for _ in range(20):
        cortex.think()
        for t in cortex.thoughts:
            if "حدس من الأحلام" in t:
                print(f"✅ Found Intuition: {t}")
                triggered = True
                break
        if triggered: break

    if not triggered:
        print("⚠️ Note: No strong intuition triggered in 20 cycles (stochastic). Run again.")

    print("\n--- Dreaming Verification Complete ---")

if __name__ == "__main__":
    verify_dreaming()
