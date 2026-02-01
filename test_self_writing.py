#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for Al-Qalam Pure.
"""

import sys
import os

# Ensure the current directory is in sys.path
sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel

def verify():
    print("Starting Al-Qalam Pure Verification...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    
    # 1. Test Concept Learning
    print("Testing Concept Learning...")
    model.learn_concept("Human", ["Ahmed", "Khalid", "Zaid"])
    
    # 2. Check if file was written
    cells_file = model.paths["cells_source"]
    if os.path.exists(cells_file):
        print(f"Success: {cells_file} created.")
        with open(cells_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "cell_concept_human" in content and "cell_inst_human_0" in content:
                print("Success: Generated code contains expected cell IDs.")
            else:
                print("Failure: Generated code is missing expected cells.")
                sys.exit(1)
    else:
        print(f"Failure: {cells_file} was not created.")
        sys.exit(1)
        
    # 3. Test Reloading
    print("Testing State Reloading...")
    new_model = SelfWritingModel(base_dir)
    if "concept_human" in new_model.cells:
        print("Success: Model successfully reloaded cells from generated source.")
    else:
        print("Failure: Model failed to reload cells.")
        sys.exit(1)
        
    print("\n[VERIFICATION PASSED]")

if __name__ == "__main__":
    verify()
