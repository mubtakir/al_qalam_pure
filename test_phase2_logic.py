#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for Al-Qalam Pure - Phase 2 (Syllogistic Inference).
"""

import sys
import os
import shutil

# Ensure the current directory is in sys.path
sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel

def verify_phase2():
    print("Starting Al-Qalam Pure Phase 2 Verification...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Cleanup previous generated files for a clean test
    auto_gen_dir = os.path.join(base_dir, "vault", "auto_generated")
    if os.path.exists(auto_gen_dir):
        shutil.rmtree(auto_gen_dir)
    os.makedirs(auto_gen_dir)

    model = SelfWritingModel(base_dir)
    
    # 1. Setup Concept and Instances
    print("Setting up Concept: Human...")
    model.learn_concept("Human", ["Khalid", "Zaid", "Ahmed"])
    
    # 2. Add an 'Apple' cell manually for facts (as an object)
    from core.dynamic_cell import DynamicCell
    apple_cell = DynamicCell("item_apple", "item", 0.0, {"name": "Apple"})
    model.cells["item_apple"] = apple_cell
    model.persist_cells()
    
    # 3. Add Facts: Khalid and Zaid like Apple
    print("Adding facts: Khalid likes Apple, Zaid likes Apple...")
    model.add_fact("Khalid", "likes", "Apple")
    model.add_fact("Zaid", "likes", "Apple")
    
    # 4. Trigger Induction
    print("Triggering Induction (Inductive Learning)...")
    model.induce_rules()
    
    # 5. Verify Rule Generation
    rules_file = model.paths["rules_source"]
    if os.path.exists(rules_file) and os.path.getsize(rules_file) > 100:
        print(f"Success: {rules_file} contains generated rules.")
    else:
        print("Failure: No rules generated.")
        sys.exit(1)
        
    # 6. Test Deduction: Ahmed should now be connected to Apple via the rule
    print("Testing Deduction (Applying self-written rules)...")
    ahmed_cell = [c for c in model.cells.values() if c.metadata.get("value") == "Ahmed"][0]
    
    # Ahmed should NOT be connected yet
    if "item_apple" in ahmed_cell.connections:
        print("Failure: Ahmed already connected to Apple before rule application.")
        sys.exit(1)
        
    # Apply rules
    context = {}
    if model.infer(context):
        print("Success: Rule triggered and applied.")
        if "item_apple" in ahmed_cell.connections:
            print("Ahmed successfully connected to Apple via self-generated logic!")
        else:
            print("Failure: Rule ran but did not establish the expected connection.")
            sys.exit(1)
    else:
        print("Failure: Rules did not trigger for Ahmed.")
        sys.exit(1)
        
    print("\n[PHASE 2 VERIFICATION PASSED]")

if __name__ == "__main__":
    verify_phase2()
