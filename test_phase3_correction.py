#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification script for Al-Qalam Pure - Phase 3 (Self-Correction).
"""

import sys
import os
import shutil

# Ensure the current directory is in sys.path
sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel

def verify_phase3():
    print("Starting Al-Qalam Pure Phase 3 Verification...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    
    # 1. Ensure we have a rule to test on
    if not model.rules:
        print("Setup: Learning concepts and inducing rules for test...")
        model.learn_concept("Human", ["UserA", "UserB", "UserC"])
        from core.dynamic_cell import DynamicCell
        if "item_gold" not in model.cells:
            model.cells["item_gold"] = DynamicCell("item_gold", "item", 0.0, {"name": "Gold"})
            model.persist_cells()
        model.add_fact("UserA", "likes", "Gold")
        model.add_fact("UserB", "likes", "Gold")
        model.induce_rules()
    
    rule_to_test = model.rules[0].__name__
    meta_before = getattr(model.rules[0], '_metadata', {})
    conf_before = meta_before.get('confidence', 0.8)
    print(f"Rule to test: {rule_to_test} (Conf before: {conf_before})")
    
    # 2. Apply Negative Feedback
    print("Applying negative feedback...")
    model.apply_feedback(rule_to_test, False)
    
    # 3. Verify Reloaded State
    new_model = SelfWritingModel(base_dir)
    refreshed_rule = [r for r in new_model.rules if r.__name__ == rule_to_test][0]
    meta_after = getattr(refreshed_rule, '_metadata', {})
    conf_after = meta_after.get('confidence', 0.0)
    
    print(f"Conf after feedback: {conf_after}")
    
    if conf_after < conf_before:
        print("Success: Confidence score updated in the source code!")
    else:
        print("Failure: Confidence score was not modified.")
        sys.exit(1)
        
    print("\n[PHASE 3 VERIFICATION PASSED]")

if __name__ == "__main__":
    verify_phase3()
