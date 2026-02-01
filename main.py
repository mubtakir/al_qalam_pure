#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Entry Point for Al-Qalam Pure.
"""

import sys
import os

# Ensure the current directory is in sys.path
sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel

def main():
    print("="*60)
    print("🧠 AL-QALAM PURE: THE Primordial REBIRTH")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    
    print(f"[System] Active Cells: {len(model.cells)}")
    print(f"[System] Active Rules: {len(model.rules)}")
    
    while True:
        print("\n[Menu]")
        print("1. Learn New Concept")
        print("2. Add Fact (Connection)")
        print("3. Consolidate Logic (Induction)")
        print("4. List Active Cells & Rules")
        print("5. Test Inference & Provide Feedback")
        print("6. Run Auditor (Contradiction Check)")
        print("7. Exit")
        
        choice = input("\nSelect an option: ").strip()
        
        if choice == "1":
            name = input("Concept Name (e.g., Human): ").strip()
            examples = input("Example instances (comma separated): ").split(",")
            examples = [ex.strip() for ex in examples if ex.strip()]
            
            if name and examples:
                model.learn_concept(name, examples)
            else:
                print("[Error] Invalid input.")
                
        elif choice == "2":
            subj = input("Subject (instance value, e.g., Khalid): ").strip()
            pred = input("Predicate (e.g., likes): ").strip()
            obj = input("Object (e.g., Apple): ").strip()
            model.add_fact(subj, pred, obj)
            
        elif choice == "3":
            model.induce_rules()
            
        elif choice == "4":
            print("\n--- Active Cells ---")
            for cid, cell in model.cells.items():
                print(f"[{cell.type.upper()}] {cid} (Value: {cell.metadata.get('name') or cell.metadata.get('value')})")
            print("\n--- Active Rules ---")
            for rule in model.rules:
                meta = getattr(rule, '_metadata', {})
                print(f"[RULE] {rule.__name__} (Conf: {meta.get('confidence', 'N/A')})")
                
        elif choice == "5":
            print("Applying inferences...")
            context = {}
            triggered = model.infer(context)
            if triggered:
                print(f"Rules triggered: {', '.join(triggered)}")
                # Feedback loop
                for rule_name in triggered:
                    fb = input(f"Was the inference from '{rule_name}' correct? (y/n): ").strip().lower()
                    if fb in ['y', 'n']:
                        model.apply_feedback(rule_name, fb == 'y')
            else:
                print("No rules triggered.")
                
        elif choice == "6":
            warnings = model.auditor.check_contradictions()
            if warnings:
                print("\n[Auditor Warnings]")
                for w in warnings:
                    print(f" - {w}")
            else:
                print("No contradictions detected.")

        elif choice == "7":
            print("Shutting down Al-Qalam Pure...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
