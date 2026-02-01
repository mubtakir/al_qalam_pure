import os
from core.self_writing_model import SelfWritingModel

def test_new_knowledge():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    
    # 1. Learn Engineer concept
    print("[1] Learning Concept: Engineer...")
    model.learn_concept("Engineer", ["Ali", "Hassan", "Omar"])
    
    # 2. Add connection facts (Ali and Hassan design Bracket)
    # Note: learn_concept creates instances with metadata 'value'
    print("[2] Adding facts...")
    # First ensure the item 'Bracket' exists by adding it to a cell if not present
    # Or just add facts which will fail if cells not found, 
    # but add_fact searches metadata.
    
    # Hack: ensure item_bracket exists
    from core.dynamic_cell import DynamicCell
    if "item_bracket" not in model.cells:
        bracket_cell = DynamicCell("item_bracket", "item", 0.0, {"name": "Bracket"})
        model.cells["item_bracket"] = bracket_cell
        model.persist_cells()

    model.add_fact("Ali", "designs", "Bracket")
    model.add_fact("Hassan", "designs", "Bracket")
    
    # 3. Induce rules
    print("[3] Inducing rules...")
    model.induce_rules()
    
    # 4. Apply feedback to the new rule
    rule_name = "rule_induct_concept_engineer_item_bracket"
    print(f"[4] Applying positive feedback to {rule_name}...")
    model.apply_feedback(rule_name, positive=True)
    
    print("\n--- Knowledge Integration Test Complete ---")
    print("Check vault/auto_generated/cells.py for updated deltas.")

if __name__ == "__main__":
    test_new_knowledge()
