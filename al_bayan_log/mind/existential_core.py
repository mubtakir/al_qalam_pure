# ==============================================================================
# BASERAH AI - EXISTENTIAL MIND (العقل الوجودي)
# ==============================================================================
# Location: bayan_core/existential_core.py
# Purpose: Analyze if entities act according to their nature
# ==============================================================================

from typing import Dict, Any, List, Optional
import re

class EntityOntology:
    """
    Ontology of entity types and their capabilities.
    قاعدة بيانات الكيانات وقدراتها الطبيعية.
    """
    
    # Entity Categories and Their Capabilities
    ONTOLOGY = {
        # === Living Entities (الكائنات الحية) ===
        "human": {
            "category": "animate",
            "subcategory": "rational",
            "can": ["eat", "sleep", "think", "feel", "speak", "walk", "run", "dream", "learn", "love", "hate"],
            "cannot": ["fly_unaided", "breathe_underwater", "photosynthesize"],
            "has": ["body", "mind", "emotions", "consciousness"],
            "arabic": ["إنسان", "رجل", "امرأة", "طفل", "شخص"]
        },
        "animal": {
            "category": "animate",
            "subcategory": "non-rational",
            "can": ["eat", "sleep", "move", "reproduce", "feel_pain"],
            "cannot": ["think_abstractly", "speak_language", "write"],
            "has": ["body", "instincts"],
            "arabic": ["حيوان", "كلب", "قط", "أسد", "طائر"]
        },
        "plant": {
            "category": "animate",
            "subcategory": "vegetative",
            "can": ["grow", "reproduce", "photosynthesize", "die"],
            "cannot": ["move", "think", "feel", "speak"],
            "has": ["roots", "leaves"],
            "arabic": ["نبات", "شجرة", "زهرة", "عشب"]
        },
        
        # === Inanimate Objects (الجمادات) ===
        "stone": {
            "category": "inanimate",
            "subcategory": "solid",
            "can": ["fall", "break", "exist", "be_moved"],
            "cannot": ["eat", "think", "feel", "move_itself", "speak", "grow"],
            "has": ["mass", "volume", "hardness"],
            "arabic": ["حجر", "صخرة"]
        },
        "water": {
            "category": "inanimate",
            "subcategory": "liquid",
            "can": ["flow", "freeze", "boil", "evaporate"],
            "cannot": ["eat", "think", "feel", "speak"],
            "has": ["volume", "temperature"],
            "arabic": ["ماء", "مياه"]
        },
        "fire": {
            "category": "inanimate",
            "subcategory": "plasma",
            "can": ["burn", "spread", "heat", "light"],
            "cannot": ["eat", "think", "feel", "freeze"],
            "has": ["temperature", "light"],
            "arabic": ["نار", "لهب"]
        },
        "machine": {
            "category": "inanimate",
            "subcategory": "artificial",
            "can": ["operate", "break", "be_repaired"],
            "cannot": ["eat", "feel", "truly_think", "dream"],
            "has": ["parts", "function"],
            "arabic": ["آلة", "سيارة", "حاسوب", "روبوت"]
        },
        
        # === Abstract Entities (المعنويات) ===
        "number": {
            "category": "abstract",
            "subcategory": "mathematical",
            "can": ["be_added", "be_multiplied", "represent_quantity"],
            "cannot": ["eat", "sleep", "move", "feel", "think", "speak", "die"],
            "has": ["value"],
            "arabic": ["رقم", "عدد", "خمسة", "عشرة"]
        },
        "idea": {
            "category": "abstract",
            "subcategory": "conceptual",
            "can": ["spread", "evolve", "influence", "be_understood"],
            "cannot": ["eat", "run", "sleep", "feel_physical", "die_physically"],
            "has": ["meaning", "content"],
            "arabic": ["فكرة", "مفهوم", "نظرية"]
        },
        "emotion": {
            "category": "abstract",
            "subcategory": "psychological",
            "can": ["be_felt", "influence_behavior", "change"],
            "cannot": ["eat", "walk", "speak_itself"],
            "has": ["intensity", "valence"],
            "arabic": ["شعور", "حب", "كره", "خوف", "فرح"]
        },
        "time": {
            "category": "abstract",
            "subcategory": "dimensional",
            "can": ["pass", "be_measured"],
            "cannot": ["eat", "think", "be_stopped", "go_backward"],
            "has": ["duration", "direction"],
            "arabic": ["وقت", "زمن", "ساعة", "يوم"]
        }
    }
    
    # Action mappings (Arabic actions map to English base)
    ACTIONS = {
        # English actions
        "eat": ["eat", "ate", "eating", "eats", "devour", "consume", "يأكل", "أكل", "آكل"],
        "sleep": ["sleep", "slept", "sleeping", "sleeps", "nap", "rest", "ينام", "نام", "نائم"],
        "think": ["think", "thought", "thinking", "thinks", "ponder", "consider", "يفكر", "فكر", "تفكير"],
        "feel": ["feel", "felt", "feeling", "feels", "sense", "يشعر", "شعر", "شعور"],
        "speak": ["speak", "spoke", "speaking", "speaks", "say", "said", "talk", "يتكلم", "تكلم", "قال"],
        "walk": ["walk", "walked", "walking", "walks", "يمشي", "مشى", "مشي"],
        "run": ["run", "ran", "running", "runs", "يركض", "ركض", "جرى", "يجري"],
        "fly_unaided": ["fly", "flew", "flying", "flies", "يطير", "طار"],
        "die": ["die", "died", "dying", "dies", "death", "يموت", "مات"],
        "grow": ["grow", "grew", "growing", "grows", "ينمو", "نما"],
        "move": ["move", "moved", "moving", "moves", "يتحرك", "تحرك"],
        "boil": ["boil", "boils", "boiling", "boiled", "يغلي", "غلى"],
    }
    
    def get_entity_type(self, text: str) -> Optional[str]:
        """Identify entity type from text."""
        text_lower = text.lower()
        
        for entity_type, info in self.ONTOLOGY.items():
            # Check English
            if entity_type in text_lower:
                return entity_type
            # Check Arabic
            for ar_name in info.get("arabic", []):
                if ar_name in text:
                    return entity_type
        
        # Fallback: try to infer from common words
        if any(w in text_lower for w in ["person", "man", "woman", "boy", "girl", "i ", "he ", "she "]):
            return "human"
        if any(w in text_lower for w in ["dog", "cat", "bird", "lion"]):
            return "animal"
        if any(w in text for w in ["أنا", "هو", "هي"]):
            return "human"
            
        return None
    
    def get_action(self, text: str) -> Optional[str]:
        """Identify action from text."""
        text_lower = text.lower()
        words = text_lower.split() + text.split()  # Both for Arabic
        
        # First pass: exact word match
        for action, variants in self.ACTIONS.items():
            for variant in variants:
                if variant in words:
                    return action
        
        # Second pass: substring match (fallback)
        for action, variants in self.ACTIONS.items():
            for variant in variants:
                if len(variant) > 3 and (variant in text_lower or variant in text):
                    return action
        
        return None
    
    def can_entity_do(self, entity_type: str, action: str) -> Dict[str, Any]:
        """Check if entity can perform action."""
        if entity_type not in self.ONTOLOGY:
            return {"valid": None, "reason": "Unknown entity type"}
        
        entity = self.ONTOLOGY[entity_type]
        
        # Normalize action
        base_action = action
        for act, variants in self.ACTIONS.items():
            if action in variants:
                base_action = act
                break
        
        # Check capabilities
        if base_action in entity.get("can", []):
            return {
                "valid": True,
                "reason": f"{entity_type} can {base_action} by nature",
                "category": entity["category"]
            }
        
        if base_action in entity.get("cannot", []):
            return {
                "valid": False,
                "reason": f"{entity_type} cannot {base_action} - not in its nature",
                "category": entity["category"],
                "violation": "Category Error"
            }
        
        # Not explicitly defined
        return {
            "valid": None,
            "reason": f"Uncertain if {entity_type} can {base_action}",
            "category": entity["category"]
        }


class ExistentialMind:
    """
    The Existential Mind - analyzes if entities act according to their nature.
    العقل الوجودي - يحلل هل الكيانات تتصرف وفق طبيعتها.
    """
    
    def __init__(self):
        self.ontology = EntityOntology()
        print("   [ExistentialMind] Ontology loaded with", len(self.ontology.ONTOLOGY), "entity types")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for existential violations.
        """
        result = {
            "input": text,
            "entity_detected": None,
            "action_detected": None,
            "analysis": "Pending",
            "valid": None,
            "violation": None,
            "depth": "Surface Level"
        }
        
        # Detect entity
        entity = self.ontology.get_entity_type(text)
        result["entity_detected"] = entity
        
        # Detect action
        action = self.ontology.get_action(text)
        result["action_detected"] = action
        
        if not entity:
            result["analysis"] = "No recognizable entity found"
            result["valid"] = None
            return result
        
        if not action:
            result["analysis"] = "No recognizable action found"
            result["valid"] = None
            return result
        
        # Check if entity can do action
        check = self.ontology.can_entity_do(entity, action)
        
        result["valid"] = check["valid"]
        result["analysis"] = check["reason"]
        result["depth"] = "Ontological Analysis"
        
        if check["valid"] == False:
            result["violation"] = check.get("violation", "Existential Violation")
        
        return result
    
    def is_valid(self, text: str) -> bool:
        """Quick check if text is existentially valid."""
        result = self.analyze(text)
        return result["valid"] != False  # None is considered acceptable


# ==============================================================================
# DEMO
# ==============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("EXISTENTIAL MIND - DEMO")
    print("=" * 50)
    
    mind = ExistentialMind()
    
    tests = [
        "The man ate a sandwich",
        "The stone ate a sandwich",
        "The number 5 slept",
        "The idea ran quickly",
        "الإنسان يأكل",
        "الحجر يأكل",
        "الرقم نام",
        "Water boils",
        "Time thinks",
    ]
    
    for t in tests:
        r = mind.analyze(t)
        status = "✅" if r["valid"] else ("❌" if r["valid"] == False else "❓")
        print(f"\n{status} '{t}'")
        print(f"   Entity: {r['entity_detected']}, Action: {r['action_detected']}")
        print(f"   Analysis: {r['analysis']}")
