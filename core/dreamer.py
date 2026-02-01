import random
from typing import List, Dict, Any, Tuple, Optional
from core.self_writing_model import SelfWritingModel

class Dreamer:
    """Explores hypothetical knowledge states to find stable patterns."""
    
    def __init__(self, model: SelfWritingModel):
        self.model = model
        self.current_dream: Optional[str] = None
        self.hypotheses: List[Dict[str, Any]] = []

    def sleep_and_dream(self) -> str:
        """Runs a single simulation cycle and returns the scenario description."""
        self.current_dream = self._generate_scenario()
        return self.current_dream

    def _generate_scenario(self) -> str:
        """Creates a 'What If' by linking random existing concepts and items."""
        concepts = [c for c in self.model.cells.values() if c.type == "concept"]
        items = [c for c in self.model.cells.values() if c.type == "item"]
        
        if not concepts or not items:
            return "الجمل في حالة سكون (لا توجد مفاهيم كافية للحلم)."

        c = random.choice(concepts)
        i = random.choice(items)
        
        c_name = c.metadata.get("name", c.id)
        i_name = i.metadata.get("name", i.id)
        
        relations = ["يحب", "يصمم", "يرتبط بـ", "يسكن في"]
        rel = random.choice(relations)
        
        scenario = f"ماذا لو كان {c_name} {rel} {i_name}؟"
        
        # Internal Weight Permutation Simulation
        # Simulate stability: if high activation across instances, it's a 'Strong Hypothesis'
        stability_score = random.random() # Placeholder for actual graph-based stability
        
        if stability_score > 0.8:
            self.hypotheses.append({
                "concept": c_name,
                "relation": rel,
                "target": i_name,
                "stability": stability_score
            })
            
        return scenario

    def get_strongest_hypothesis(self) -> Optional[str]:
        """Returns the most stable 'Dreamed' connection for Cortex verification."""
        if not self.hypotheses:
            return None
        
        best = max(self.hypotheses, key=lambda x: x["stability"])
        self.hypotheses = [] # Clear after retrieval
        return f"حدس من الأحلام: يبدو منطقياً أن {best['concept']} {best['relation']} {best['target']}. هل نعتمد هذا؟"
