# Al-Qalam Pure: Dynamic Rules Registry


# --- INFERENCE RULE: induct_concept_human_item_apple ---
# Description: If an entity is a Human, it likely relates to Apple
def rule_induct_concept_human_item_apple(model, context):
    """Generated logical rule: If an entity is a Human, it likely relates to Apple"""
    
    # Condition Logic
    def check_condition(model, context):
        targets = []
        for cid in model.cells:
            cell = model.cells[cid]
            if cell.type == 'instance' and 'concept_human' in cell.connections:
                if 'item_apple' not in cell.connections:
                    # Check confidence
                    conf = getattr(rule_induct_concept_human_item_apple, '_metadata', {}).get('confidence', 1.0)
                    if conf > 0.3: # Only trigger if confident
                        targets.append(cell)
        if targets:
            context['targets'] = targets
            return True
        return False
        
    # Action Logic
    def perform_action(model, context):
        for target in context['targets']:
            target.connect_to('item_apple', 0.8)
            print(f"[Inference] Rule applied: {target.metadata.get('value')} now connected to item_apple")
        model.persist_cells()
    
    if check_condition(model, context):
        perform_action(model, context)
        return True
    return False

# Registering metadata for induct_concept_human_item_apple
rule_induct_concept_human_item_apple._metadata = {'confidence': 0.8, 'created': '2026-02-01T06:58:44.352133'}
