from typing import Dict, Any, Optional
from .node import BayanNode

class CausalRelation:
    """
    Represents a directed causal link between two nodes.
    Supported types: causes, enables, prevents, requires, leads_to, etc.
    """
    def __init__(self, source: BayanNode, target: BayanNode, rel_type: str, strength: float = 1.0, conditions: Dict[str, Any] = None):
        self.source = source
        self.target = target
        self.type = rel_type
        self.strength = max(0.0, min(1.0, float(strength))) # Clamp 0-1
        self.conditions = conditions or {}
        
    def is_active(self, context: Dict[str, Any] = None) -> bool:
        """
        Checks if this relation is currently active based on conditions.
        Future: Implement physical condition checking (e.g. distance < X).
        """
        # Placeholder for complex condition logic
        if not self.conditions:
            return True
        
        # Example simple check
        if context:
            for key, required_val in self.conditions.items():
                if context.get(key) != required_val:
                    return False
        return True

    def __repr__(self):
        return f"({self.source.id}) --[{self.type}:{self.strength}]--> ({self.target.id})"
