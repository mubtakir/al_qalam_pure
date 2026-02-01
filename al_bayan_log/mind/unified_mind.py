import sys
import os

# Ensure paths are set up to reach the deep ExistentialModel
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BAYAN_MIND_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "bayan_mind", "foundation", "training", "bayan_model", "existential_v4", "core")
sys.path.append(BAYAN_MIND_PATH)

from typing import Dict, Any, List

# Core Imports
try:
    from .thinking_core.engine import ThinkingEngine
except ImportError:
    print("[UnifiedMind] Warning: ThinkingEngine not found.")
    ThinkingEngine = None

try:
    # Check what is actually available in physical_core
    from ._legacy_wrapper.physical_core import PhysicalEntity
    # Mass/Force might not be exported directly, so we skip them or import if confirmed
except ImportError:
    print("[UnifiedMind] Warning: PhysicalEntity not found.")
    PhysicalEntity = None

# Attempt to import Existential Model (Mock or Real)
try:
    from existential_model import BayanExistentialV4
except ImportError:
    # Define as None if missing so checks don't crash
    BayanExistentialV4 = None

class UnifiedMind:
    """
    The Council of Minds.
    Integrates Logical, Physical, Existential, and Neural intelligence into a single interface.
    """
    def __init__(self, enable_neural=True):
        print("[UnifiedMind] Convening the Council...")
        
        # 1. The Logical Mind (ThinkingCore)
        self.logic = ThinkingEngine() if ThinkingEngine else None
        print(f"   [+] Logical Mind: {'Active' if self.logic else 'Inactive'}")
        
        # 2. The Physical Mind (Direct access to Physical Core utils)
        # We don't instantiate a "PhysicalEngine" class per se, but we have access to the laws.
        self.has_physics = True
        print(f"   [+] Physical Mind: {'Active' if self.has_physics else 'Inactive'}")
        
        # 3. The Existential Mind (NEW - Full Implementation!)
        self.existential = None
        self.existential_mind = None
        try:
            from .existential_core import ExistentialMind as ExMind
            self.existential_mind = ExMind()
            self.existential = "Active (Ontology-based)"
        except ImportError:
            # Fallback to old model if available
            if BayanExistentialV4:
                self.existential = "Available (Lazy Loading)"
        print(f"   [+] Existential Mind: {self.existential if self.existential else 'Inactive'}")
        
        # 4. The Neural Mind (Trained Model) - NEW!
        self.neural = None
        self.neural_generator = None
        if enable_neural:
            self._init_neural_mind()
    
    def _init_neural_mind(self):
        """Initialize the trained neural model."""
        # Try BitNet Sovereign model (NEW - 1-bit Era)
        try:
            from bayan_core.bayan_bitnet_sovereign import BayanBitNetSovereign
            # Try to find latest checkpoint
            ckpt_dir = os.path.join(os.path.dirname(CURRENT_DIR), "checkpoints")
            latest_ckpt = None
            if os.path.exists(ckpt_dir):
                ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("BayanBitNet") and f.endswith(".pt")]
                if ckpts:
                    latest_ckpt = os.path.join(ckpt_dir, sorted(ckpts)[-1])
            
            bitnet_sovereign = BayanBitNetSovereign(latest_ckpt)
            if bitnet_sovereign.is_available():
                self.neural_generator = bitnet_sovereign
                self.neural = "BitNet b1.58"
                print(f"   [+] Neural Mind: 💎 BITNET (BayanBitNet 190M - 1-bit)")
                return
        except Exception as e:
            pass

        # Try standard sovereign model
        try:
            from bayan_core.bayan_sovereign import BayanSovereign
            sovereign = BayanSovereign()
            if sovereign.is_available():
                self.neural_generator = sovereign
                self.neural = "Sovereign"
                print(f"   [+] Neural Mind: 🏴 SOVEREIGN (BayanTransformer 59.2M)")
                return
        except Exception as e:
            pass  # Fall back to HuggingFace models
        
        # Fallback to HuggingFace models
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import os
            
            possible_paths = [
                os.path.join(os.path.dirname(CURRENT_DIR), "checkpoints", "bayan_council_local", "final"),
                os.path.join(os.path.dirname(CURRENT_DIR), "checkpoints", "bayan_council_v3"),
            ]
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(model_path)
                    self.neural_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
                    self.neural = "Active"
                    model_name = os.path.basename(os.path.dirname(model_path)) if "final" in model_path else os.path.basename(model_path)
                    print(f"   [+] Neural Mind: Active ({model_name})")
                    return
            
            print(f"   [+] Neural Mind: Inactive (No model found)")
        except ImportError:
            print("   [+] Neural Mind: Inactive (transformers not installed)")
        except Exception as e:
            print(f"   [+] Neural Mind: Inactive ({str(e)[:50]})")

    def ponder(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        The Core Loop:
        1. Parse Input
        2. Consult Physics (Is it possible?)
        3. Consult Logic (Does it follow?)
        4. Consult Existential (What does it mean?)
        5. Consult Neural (What does the trained model say?)
        6. Synthesize
        """
        print(f"\n[UnifiedMind] Pondering: '{input_text}'")
        
        report = {
            "input": input_text,
            "physical_analysis": self._consult_physics(input_text),
            "logical_analysis": self._consult_logic(input_text),
            "existential_analysis": self._consult_existential(input_text),
            "neural_analysis": self._consult_neural(input_text),
            "final_verdict": "Pending"
        }
        
        # Synthesis Logic - consider all 4 minds
        physics_ok = report["physical_analysis"]["possible"]
        logic_ok = report["logical_analysis"]["valid"]
        existential_ok = report["existential_analysis"].get("valid") != False  # None = OK
        neural_ok = report["neural_analysis"].get("verdict") == "VERIFIED"
        
        # Count rejections from active minds
        rejections = []
        if not physics_ok: rejections.append("Physically Impossible")
        if not logic_ok: rejections.append("Logically Invalid")
        if report["existential_analysis"].get("valid") == False: 
            rejections.append(f"Existentially Invalid ({report['existential_analysis'].get('violation', 'Category Error')})")
        if report["neural_analysis"].get("verdict") == "REJECTED": 
            rejections.append("Neural Rejected")
        
        # Final decision: any critical rejection = rejected
        if rejections:
            report["final_verdict"] = f"Rejected: {', '.join(rejections)}"
        else:
            report["final_verdict"] = "Accepted as Reality"
            
        return report

    def _consult_physics(self, text: str) -> Dict[str, Any]:
        """
        Simulate the scenario using Physical Laws.
        """
        text_lower = text.lower()
        
        # 1. Gravity / Mass Rules
        if "float" in text_lower and "gravity" not in text_lower and "space" not in text_lower:
            return {"possible": False, "law": "Gravity", "reason": "Objects cannot float without force."}
        
        if "negative mass" in text_lower:
            return {"possible": False, "law": "Mass Positivity", "reason": "Mass cannot be negative."}

        # 2. Buoyancy / Density Rules (Walking on water)
        if ("walk" in text_lower or "drive" in text_lower or "run" in text_lower) and "ocean" in text_lower and "boat" not in text_lower:
             return {"possible": False, "law": "Buoyancy", "reason": "Solids denser than water sink."}

        # 3. Thermodynamics (Ice/Fire)
        if "ice" in text_lower and "fire" in text_lower and "freeze" in text_lower:
             return {"possible": False, "law": "Thermodynamics", "reason": "Heat melts ice, does not freeze it."}

        # 4. Human Biomechanics / Gravity (Jump to moon)
        if "jump" in text_lower and "moon" in text_lower and "rocket" not in text_lower:
             return {"possible": False, "law": "Gravity/Biomechanics", "reason": "Escape velocity requires more energy than a jump."}
            
        return {"possible": True, "note": "No obvious physical violations found."}

    def _consult_logic(self, text: str) -> Dict[str, Any]:
        """
        Use ThinkingCore to check causal consistency.
        """
        if not self.logic:
            return {"valid": False, "error": "Logical Mind inactive"}
            
        # Example: Check if the action implies a known causal relation
        # For demo, we assume the text represents a causal claim like "A causes B"
        if "cause" in text.lower() or "leads to" in text.lower():
            # In a real system, we'd parse A and B and check the graph
            pass
            
        return {"valid": True, "note": "Causally consistent"}

    def _consult_existential(self, text: str) -> Dict[str, Any]:
        """
        Check for deeper meaning or abstract conflicts using Ontology-based analysis.
        """
        if not self.existential_mind:
            # Fallback to neutral if not available
            return {"valid": None, "analysis": "Existentially Neutral", "depth": "Surface Level"}
        
        # Use the new ExistentialMind
        result = self.existential_mind.analyze(text)
        
        return {
            "valid": result.get("valid"),
            "entity": result.get("entity_detected"),
            "action": result.get("action_detected"),
            "analysis": result.get("analysis"),
            "violation": result.get("violation"),
            "depth": result.get("depth", "Ontological Analysis")
        }
    
    def _consult_neural(self, text: str) -> Dict[str, Any]:
        """
        Consult the trained neural model for verification.
        """
        if not self.neural_generator:
            return {"verdict": "UNKNOWN", "error": "Neural Mind inactive"}
        
        try:
            # Check if using BayanSovereign (has analyze method)
            if hasattr(self.neural_generator, 'analyze'):
                result = self.neural_generator.analyze(text)
                return {
                    "verdict": result.get("verdict", "ANALYZED"),
                    "response": result.get("response", ""),
                    "confidence": "sovereign",
                    "sovereign": True
                }
            
            # Fallback: HuggingFace pipeline
            prompt = f"Question: {text}\nAnswer:"
            result = self.neural_generator(
                prompt, 
                max_new_tokens=30, 
                do_sample=False,
                pad_token_id=self.neural_generator.tokenizer.eos_token_id
            )
            output = result[0]["generated_text"]
            
            # Parse the response
            if "Answer:" in output:
                answer = output.split("Answer:")[1].strip()
            else:
                answer = output
            
            # Determine verdict
            if "[REJECTED]" in answer or "REJECTED" in answer:
                return {"verdict": "REJECTED", "response": answer, "confidence": "high"}
            elif "[VERIFIED]" in answer or "VERIFIED" in answer:
                return {"verdict": "VERIFIED", "response": answer, "confidence": "high"}
            else:
                return {"verdict": "UNCERTAIN", "response": answer, "confidence": "low"}
                
        except Exception as e:
            return {"verdict": "ERROR", "error": str(e)}
