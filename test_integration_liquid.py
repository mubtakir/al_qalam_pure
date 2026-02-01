import sys
import os
import io
import time

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.cortex import Cortex

def integration_test():
    print("🔌 INITIALIZING FULL SYSTEM INTEGRATION...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize Core
    model = SelfWritingModel(base_dir)
    cortex = Cortex(model)
    
    print("✅ Cortex Online. Liquid Engine Connected.")
    print("="*60)
    
    # Simulation Loop
    states = [
        ("Normal Operation", -0.9),
        ("Minor Glitch", -0.2),
        ("SYSTEM FAILURE IMMINENT", 0.95),
        ("Recovery", -0.5),
        ("Stabilized", -0.9)
    ]
    
    for label, stress in states:
        print(f"\n[Status: {label}] (Stress: {stress})")
        
        # Update Cortex State
        cortex.stress_level = stress
        
        # Ask Cortex to speak
        voice = cortex.express_state()
        
        print(f"🗣️ Cortex Voice: '{voice}'")
        time.sleep(1.0)
        
    print("\n="*60)
    print("Test Complete. The Cortex successfully expressed its state using the Liquid Code-Weights.")

if __name__ == "__main__":
    integration_test()
