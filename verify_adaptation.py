import os
import sys
from core.self_writing_model import SelfWritingModel

def verify():
    print("--- Verification: Dynamic Executable Weights ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    
    # 1. Force a persistence to update cells.py to new format
    print("[1] Persisting cells to new format...")
    model.persist_cells()
    
    # 2. Check cells.py content
    cells_path = model.paths["cells_source"]
    with open(cells_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if "delta=" in content and "# adaptive_slot" in content:
            print("[Success] cells.py updated with adaptive slots.")
        else:
            print("[Failure] cells.py format incorrect.")
            return

    # 3. Simulate feedback to update a delta slot
    print("[2] Simulating feedback for rule_induct_concept_human_item_apple...")
    # This rule exists in my previous knowledge and should be in rules.py
    model.apply_feedback("rule_induct_concept_human_item_apple", positive=True)
    
    # 4. Check if cells.py delta was modified
    with open(cells_path, 'r', encoding='utf-8') as f:
        new_content = f.read()
        if "delta=0.10" in new_content or "delta=+0.10" in new_content:
             print("[Success] Delta slot surgically updated in cells.py!")
        else:
             # Find any delta update
             import re
             deltas = re.findall(r"delta=([\d\.\+\-]+)", new_content)
             print(f"[Info] Found deltas: {deltas}")
             if any(float(d) != 0.0 for d in deltas):
                 print("[Success] Delta slots are moving!")
             else:
                 print("[Failure] Delta slots remained static.")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify()
