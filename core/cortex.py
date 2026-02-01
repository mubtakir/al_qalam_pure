from typing import List, Dict, Any, Optional
from core.self_writing_model import SelfWritingModel
from core.dreamer import Dreamer
from core.liquid_engine.liquid_narrator import LiquidNarrator

class Cortex:
    """The central reasoning unit that orchestrates the SelfWritingModel and identifies meta-patterns."""
    
    def __init__(self, model: SelfWritingModel):
        self.model = model
        self.dreamer = Dreamer(model)
        self.thoughts: List[str] = []
        
        # Liquid Engine Integration
        self.liquid_voice = LiquidNarrator()
        self.stress_level = -0.8 # Default to Calm

    def express_state(self) -> str:
        """Uses the Liquid Engine to express internal state."""
        # Update mood based on stress
        self.liquid_voice.set_mood(self.stress_level)
        return self.liquid_voice.speak("system")

    def think(self) -> List[str]:
        """Runs periodic analysis to find gaps, proposed analogies, and dreams."""
        self.thoughts = []
        
        # 0. Process Dreams
        dream_scenario = self.dreamer.sleep_and_dream()
        # Add to thoughts if it's a strong hypothesis
        hypo = self.dreamer.get_strongest_hypothesis()
        if hypo:
            self.thoughts.append(hypo)

        # 1. Identify Knowledge Gaps
        self._check_domain_gaps()
        
        # 2. Propose Analogies
        self._propose_analogies()
        
        return self.thoughts

    def _check_domain_gaps(self):
        """Finds concepts that have significantly fewer facts than others in the same group."""
        concepts = [c for c in self.model.cells.values() if c.type == "concept"]
        if not concepts: return
        
        # Calculate average fact density per concept
        density = {}
        for c in concepts:
             instances = [self.model.cells[cid] for cid in c.connections if cid in self.model.cells and self.model.cells[cid].type == "instance"]
             fact_count = sum(len(inst.metadata.get("facts", [])) for inst in instances)
             density[c.id] = fact_count

        avg_density = sum(density.values()) / len(density)
        for cid, count in density.items():
            if count < avg_density * 0.5:
                concept_name = self.model.cells[cid].metadata.get("name", cid)
                self.thoughts.append(f"ملاحظة: مفهوم '{concept_name}' لديه بيانات أقل من المتوسط. هل تود إضافة المزيد من الحقائق عنه؟")

    def _propose_analogies(self):
        """Attempts to suggest cross-domain links based on rule patterns."""
        concepts = [c for c in self.model.cells.values() if c.type == "concept"]
        for c1 in concepts:
            name1 = c1.metadata.get("name", c1.id)
            for c2 in concepts:
                if c1.id == c2.id: continue
                name2 = c2.metadata.get("name", c2.id)
                
                # Look for rules applied to c1 but not c2
                for rule in self.model.rules:
                    doc = rule.__doc__ or ""
                    if name1 in doc and name2 not in doc:
                        # Extract the target of the link from the docstring
                        # Format: "If an entity is a [Name1], it likely relates to [Target]"
                        if "relates to " in doc:
                            target_part = doc.split("relates to ")[-1]
                            self.thoughts.append(f"قياس منطقي: بما أن {name1} يرتبط بـ {target_part}، هل تفكر أن {name2} قد يرتبط به أيضاً؟")

    def generate_meta_rules(self):
        """Merge similar rules into abstract general patterns."""
        # Placeholder for complex abstraction
        pass
