#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelfWritingModel: The orchestrator that manages cells, rules, and modifies its own source.
"""

import os
import json
import shutil
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from core.dynamic_cell import DynamicCell
from core.inference_rule import InferenceRule
from core.auditor import Auditor
from core.logical_bridge import LogicalBridge

# Bayan Safety System
try:
    from al_bayan_log.safety import ImmuneSystem
    IMMUNE_AVAILABLE = True
except ImportError:
    IMMUNE_AVAILABLE = False
    ImmuneSystem = None

# Bayan LLM Bridge (Donor Brain)
try:
    from al_bayan_log.llm import LLMBridge
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMBridge = None

# Dynamic Language Engine V3.0
try:
    from core.language_engine import GenerationEngine, DynamicVocab, DynamicGrammar
    LANG_ENGINE_AVAILABLE = True
except ImportError:
    LANG_ENGINE_AVAILABLE = False
    GenerationEngine = None

class SelfWritingModel:
    """A model that learns and persists its knowledge as executable Python code."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.cells: Dict[str, DynamicCell] = {}
        self.rules: List[Callable] = []
        self.auditor = Auditor(self)
        self.logic = LogicalBridge()  # Bayan Logical Engine Integration
        
        # Bayan ImmuneSystem for code safety
        if IMMUNE_AVAILABLE:
            self.immune = ImmuneSystem(os.path.join(base_dir, "vault", "auto_generated"))
        else:
            self.immune = None
        
        # LLM Bridge (lazy loaded)
        self._llm = None
        
        # Dynamic Language Engine V3.0
        if LANG_ENGINE_AVAILABLE:
            vocab_path = os.path.join(base_dir, "vault", "language", "vocab.py")
            grammar_path = os.path.join(base_dir, "vault", "language", "patterns.py")
            self.language = GenerationEngine(
                vocab=DynamicVocab(persist_path=vocab_path),
                grammar=DynamicGrammar(persist_path=grammar_path)
            )
        else:
            self.language = None
        
        # Paths
        self.paths = {
            "cells_source": os.path.join(base_dir, "vault", "auto_generated", "cells.py"),
            "rules_source": os.path.join(base_dir, "vault", "auto_generated", "rules.py"),
            "knowledge_base": os.path.join(base_dir, "vault", "knowledge_base.json"),
            "backups": os.path.join(base_dir, "vault", "backups"),
            "vocab": os.path.join(base_dir, "vault", "language", "vocab.py"),
            "patterns": os.path.join(base_dir, "vault", "language", "patterns.py")
        }
        
        self._ensure_paths()
        self.load_state()

    def _ensure_paths(self):
        os.makedirs(os.path.dirname(self.paths["cells_source"]), exist_ok=True)
        os.makedirs(self.paths["backups"], exist_ok=True)
        
        if not os.path.exists(self.paths["cells_source"]) or os.path.getsize(self.paths["cells_source"]) == 0:
            with open(self.paths["cells_source"], 'w', encoding='utf-8') as f:
                f.write("# Al-Qalam Pure: Dynamic Cells Registry\n")
                f.write("from core.dynamic_cell import DynamicCell\n\n")
        
        if not os.path.exists(self.paths["rules_source"]):
            with open(self.paths["rules_source"], 'w', encoding='utf-8') as f:
                f.write("# Al-Qalam Pure: Dynamic Rules Registry\n\n")

    def load_state(self):
        """Loads cells and rules from generated python files."""
        if os.path.exists(self.paths["cells_source"]):
            try:
                from core.dynamic_cell import AdaptiveParameter
                namespace = {"DynamicCell": DynamicCell, "AdaptiveParameter": AdaptiveParameter}
                with open(self.paths["cells_source"], 'r', encoding='utf-8') as f:
                    exec(f.read(), namespace)
                self.cells = {obj.id: obj for obj in namespace.values() if isinstance(obj, DynamicCell)}
            except Exception as e:
                print(f"[Error] Failed to load cells: {e}")

        if os.path.exists(self.paths["rules_source"]):
            try:
                namespace = {}
                with open(self.paths["rules_source"], 'r', encoding='utf-8') as f:
                    exec(f.read(), namespace)
                self.rules = [obj for name, obj in namespace.items() if name.startswith('rule_') and callable(obj)]
            except Exception as e:
                print(f"[Error] Failed to load rules: {e}")

    def learn_concept(self, name: str, examples: List[str]):
        """Creates a new concept and its instances, then persists them."""
        concept_id = f"concept_{name.lower().replace(' ', '_')}"
        concept_cell = DynamicCell(concept_id, "concept", 1.0, {"name": name})
        self.cells[concept_id] = concept_cell
        
        for i, ex in enumerate(examples):
            inst_id = f"inst_{name.lower().replace(' ', '_')}_{i}"
            inst_cell = DynamicCell(inst_id, "instance", 0.5, {"value": ex})
            inst_cell.connect_to(concept_id, 0.9)
            concept_cell.connect_to(inst_id, 0.9)
            self.cells[inst_id] = inst_cell
            
        self.persist_cells()
        
        # Sync to Bayan LogicalEngine
        self.logic.sync_cell(concept_cell)
        for i in range(len(examples)):
            inst_id = f"inst_{name.lower().replace(' ', '_')}_{i}"
            self.logic.sync_cell(self.cells[inst_id])

    def add_fact(self, subject: str, predicate: str, object_val: str):
        """Adds a logical fact by connecting cells."""
        sub_cell = self._find_cell_by_meta_value(subject)
        obj_cell = self._find_cell_by_meta_value(object_val)
        
        if sub_cell and obj_cell:
            sub_cell.connect_to(obj_cell.id, 1.0)
            sub_cell.metadata.setdefault("facts", []).append(f"{predicate} {object_val}")
            self.persist_cells()
            print(f"[Fact] Connected {subject} --({predicate})--> {object_val}")
        else:
            print(f"[Error] Could not find cells for {subject} or {object_val}.")

    def query_knowledge(self, query_val: str) -> str:
        """Queries knowledge about an entity using both cell network and logical engine."""
        cell = self._find_cell_by_meta_value(query_val)
        if not cell:
            return f"لا أجد معلومات عن '{query_val}'."
        
        # Build response from cell metadata
        response_parts = []
        
        # Basic info
        cell_name = cell.metadata.get("name", cell.metadata.get("value", cell.id))
        response_parts.append(f"📌 **{cell_name}** (النوع: {cell.type})")
        
        # Facts from metadata
        facts = cell.metadata.get("facts", [])
        if facts:
            response_parts.append("الحقائق: " + "، ".join(facts))
        
        # Connections
        connections = []
        for target_id, param in cell.connections.items():
            if target_id in self.cells:
                target_cell = self.cells[target_id]
                target_name = target_cell.metadata.get("name", target_cell.metadata.get("value", target_id))
                weight = param.value if hasattr(param, 'value') else param
                connections.append(f"{target_name} ({weight:.1f})")
        
        if connections:
            response_parts.append(f"متصل بـ: {', '.join(connections)}")
        
        # Logical query (via Bayan engine)
        self.logic.sync_cell(cell)
        logical_connections = self.logic.query("connected", [cell.id, "?Target", "?Weight"])
        if logical_connections:
            response_parts.append(f"🔗 اتصالات منطقية: {len(logical_connections)} اتصال")
        
        return "\n".join(response_parts)

    def _find_cell_by_meta_value(self, value: str) -> Optional[DynamicCell]:
        for cell in self.cells.values():
            if cell.metadata.get("value") == value or cell.metadata.get("name") == value:
                return cell
        return None

    def induce_rules(self):
        """Analyzes patterns and generates new InferenceRules."""
        print("[Induction] Analyzing patterns...")
        concepts = [c for c in self.cells.values() if c.type == "concept"]
        for concept in concepts:
            instances = [self.cells[cid] for cid in concept.connections if cid in self.cells and self.cells[cid].type == "instance"]
            if not instances: continue
            
            common_targets = {}
            for inst in instances:
                for target_id in inst.connections:
                    if target_id == concept.id: continue
                    common_targets[target_id] = common_targets.get(target_id, 0) + 1
            
            for target_id, count in common_targets.items():
                if count >= 2:
                    target_cell = self.cells[target_id]
                    rule_id = f"induct_{concept.id}_{target_id}"
                    
                    if any(r.__name__ == f"rule_{rule_id}" for r in self.rules):
                        continue
                    
                    print(f"[Induction] New rule discovered: {concept.metadata['name']} -> {target_cell.metadata.get('name', target_id)}")
                    
                    description = f"If an entity is a {concept.metadata['name']}, it likely relates to {target_cell.metadata.get('name', target_id)}"
                    cond_code = f"""
targets = []
for cid in model.cells:
    cell = model.cells[cid]
    if cell.type == 'instance' and '{concept.id}' in cell.connections:
        if '{target_id}' not in cell.connections:
            # Check confidence
            conf = getattr(rule_{rule_id}, '_metadata', {{}}).get('confidence', 1.0)
            if conf > 0.3: # Only trigger if confident
                targets.append(cell)
if targets:
    context['targets'] = targets
    return True
return False
"""
                    action_code = f"""
for target in context['targets']:
    target.connect_to('{target_id}', 0.8)
    print(f"[Inference] Rule applied: {{target.metadata.get('value')}} now connected to {target_id}")
model.persist_cells()
"""
                    meta = {"confidence": 0.8, "created": datetime.now().isoformat()}
                    new_rule = InferenceRule(rule_id, description, cond_code, action_code, meta)
                    self._persist_rule(new_rule)

    def _persist_rule(self, rule: InferenceRule):
        with open(self.paths["rules_source"], 'a', encoding='utf-8') as f:
            f.write(rule.to_source_code())
        self.load_state()

    def apply_feedback(self, rule_name: str, positive: bool):
        """Adjusts the confidence of a self-written rule or its triggered connections."""
        if not os.path.exists(self.paths["rules_source"]): return
        
        delta = self.auditor.calculate_confidence_delta(positive)
        
        # 1. Update rule confidence in rules.py
        with open(self.paths["rules_source"], 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Pattern: rule_name._metadata = {'confidence': 0.8, ...}
        pattern = rf"({rule_name}\._metadata = \{{'confidence': )([\d\.]+)([^}}]*\}})"
        
        def update_conf(match):
            prefix = match.group(1)
            old_conf = float(match.group(2))
            suffix = match.group(3)
            new_conf = max(0.0, min(1.0, old_conf + delta))
            print(f"[Feedback] Updating {rule_name} confidence: {old_conf:.2f} -> {new_conf:.2f}")
            return f"{prefix}{new_conf:.2f}{suffix}"
            
        new_content = re.sub(pattern, update_conf, content)
        if new_content != content:
            with open(self.paths["rules_source"], 'w', encoding='utf-8') as f:
                f.write(new_content)

        # 2. Update dynamic weights in cells.py (Surgical modification of delta slots)
        if os.path.exists(self.paths["cells_source"]):
            with open(self.paths["cells_source"], 'r', encoding='utf-8') as f:
                cells_content = f.read()
            
            # Find connections modified by this rule (Logic: find the target cells and their connection lines)
            # This is a bit heuristic, but it targets lines with # adaptive_slot
            def update_delta_slot(match):
                prefix = match.group(1)
                old_delta = float(match.group(2))
                suffix = match.group(3)
                new_delta = old_delta + delta
                print(f"[Feedback] Updating delta slot: {old_delta:+.2f} -> {new_delta:+.2f}")
                return f"{prefix}{new_delta:+.2f}{suffix}"

            # Regex for: .connect_to('...', weight=..., delta=0.0) # adaptive_slot
            slot_pattern = r"(\.connect_to\('.*', weight=[\d\.]+, delta=)([\d\.\+\-]+)(\) # adaptive_slot)"
            new_cells_content = re.sub(slot_pattern, update_delta_slot, cells_content)
            
            if new_cells_content != cells_content:
                with open(self.paths["cells_source"], 'w', encoding='utf-8') as f:
                    f.write(new_cells_content)

        self.load_state()
        return True

    def infer(self, context: Dict) -> List[str]:
        """Applies rules and returns the names of the rules that triggered."""
        triggered_rules = []
        while True:
            applied_this_loop = False
            for rule in self.rules:
                if rule(self, context):
                    applied_this_loop = True
                    if rule.__name__ not in triggered_rules:
                        triggered_rules.append(rule.__name__)
            if not applied_this_loop:
                break
        return triggered_rules

    def persist_cells(self):
        """Writes all cells to the source file (with ImmuneSystem validation)."""
        # Generate the new code
        new_code = "# Al-Qalam Pure: Dynamic Cells Registry\n"
        new_code += "from core.dynamic_cell import DynamicCell, AdaptiveParameter\n\n"
        for cell in self.cells.values():
            new_code += cell.to_source_code()
            new_code += "\n"
        
        # Validate with ImmuneSystem before saving
        if self.immune:
            validation = self.immune.sandbox_test(new_code)
            if not validation["valid"]:
                print(f"[IMMUNE] Code rejected: {validation['error']}")
                return False
        
        # Write to file
        with open(self.paths["cells_source"], 'w', encoding='utf-8') as f:
            f.write(new_code)
        return True
    
    def safe_generate_code(self, code: str, description: str = "") -> dict:
        """
        Tests code in sandbox before accepting it.
        Returns: {"valid": bool, "error": str/None}
        """
        if not self.immune:
            return {"valid": True, "error": None, "warning": "ImmuneSystem not available"}
        
        result = self.immune.sandbox_test(code)
        if result["valid"]:
            print(f"[IMMUNE] Code approved: {description}")
        else:
            print(f"[IMMUNE] Code rejected: {result['error']}")
        return result
    
    @property
    def llm(self):
        """Lazy-loaded LLM Bridge (Donor Brain)."""
        if self._llm is None and LLM_AVAILABLE:
            try:
                self._llm = LLMBridge()
            except Exception as e:
                print(f"[LLM] Could not load: {e}")
                self._llm = False  # Mark as failed
        return self._llm if self._llm else None
    
    def ask_llm(self, question: str) -> str:
        """Ask the donor brain a question."""
        if not self.llm:
            return "[LLM غير متوفر]"
        return self.llm.think(question)
    
    def generate_code_with_llm(self, task: str, max_retries: int = 3) -> dict:
        """
        Use LLM to generate code, validate with ImmuneSystem.
        Retries with error feedback if code fails validation.
        
        Returns: {"success": bool, "code": str, "attempts": int}
        """
        if not self.llm:
            return {"success": False, "code": None, "error": "LLM not available", "attempts": 0}
        
        prompt = f"Generate Python code for: {task}\n\nReturn ONLY code inside ```python ``` blocks."
        
        for attempt in range(1, max_retries + 1):
            print(f"[LLM] Attempt {attempt}/{max_retries}...")
            
            result = self.llm.think_and_validate(prompt)
            
            if result["valid"]:
                # Double-check with our ImmuneSystem
                if self.immune:
                    check = self.immune.sandbox_test(result["code"])
                    if check["valid"]:
                        print(f"[LLM] Code generated and validated successfully!")
                        return {"success": True, "code": result["code"], "attempts": attempt}
                    else:
                        # Add error to prompt for retry
                        prompt = f"{prompt}\n\nPrevious attempt failed: {check['error']}\nPlease fix:"
                else:
                    return {"success": True, "code": result["code"], "attempts": attempt}
            else:
                prompt = f"{prompt}\n\nPrevious attempt failed: {result['error']}\nPlease fix:"
        
        return {"success": False, "code": None, "error": "Max retries exceeded", "attempts": max_retries}
    
    # === KnowledgeDigester Compatibility ===
    
    @property
    def immune_system(self):
        """Alias for self.immune (for KnowledgeDigester compatibility)."""
        return self.immune
    
    def ask_donor_brain(self, system_prompt: str, user_prompt: str, expect_code: bool = False) -> str:
        """
        Ask the LLM with custom system prompt (for KnowledgeDigester).
        """
        if not self.llm:
            return None
        return self.llm.think(user_prompt, system_prompt)
    
    def learn_logic_concept(self, concept_name: str, methods: dict) -> Optional[DynamicCell]:
        """
        Learns a logic concept from generated methods.
        Creates a concept cell and stores the methods.
        """
        # Create concept cell
        cell_id = f"logic_{concept_name.lower().replace(' ', '_')}"
        
        # Assemble the code
        code_parts = []
        for method_name, method_code in methods.items():
            code_parts.append(method_code)
        full_code = "\n\n".join(code_parts)
        
        # Validate with ImmuneSystem
        if self.immune:
            result = self.immune.sandbox_test(full_code)
            if not result["valid"]:
                print(f"[IMMUNE] Logic concept rejected: {result['error']}")
                return None
        
        # Create the cell
        cell = DynamicCell(
            cell_id=cell_id,
            cell_type="logic_concept",
            metadata={
                "name": concept_name,
                "methods": list(methods.keys()),
                "code": full_code
            }
        )
        
        self.cells[cell_id] = cell
        self.persist_cells()
        self.logic.sync_cell(cell)
        
        print(f"[LEARN] Logic concept '{concept_name}' learned with {len(methods)} methods")
        return cell
    
    def learn_lesson(self, topic: str, lesson_text: str) -> str:
        """
        High-level method to learn from a natural language lesson.
        Uses LLM to generate code, ImmuneSystem to validate, then saves.
        """
        print(f"📚 Learning lesson: {topic}")
        
        if not self.llm:
            return "❌ LLM not available"
        
        # Simple prompt for code generation
        prompt = f"""Convert this lesson into Python code:

Topic: {topic}
Lesson: {lesson_text}

Return a Python function or class that implements this knowledge.
Use ```python ``` code blocks."""
        
        result = self.generate_code_with_llm(prompt)
        
        if result["success"]:
            # Save as a logic concept
            cell = self.learn_logic_concept(topic, {"main": result["code"]})
            if cell:
                return f"✅ Lesson '{topic}' learned successfully!"
            else:
                return "❌ Code generated but failed validation"
        else:
            return f"❌ Failed to learn: {result.get('error', 'Unknown error')}"
    
    # === Dynamic Language Engine V3.0 ===
    
    def speak(self, subject: str, action: str = None, obj: str = None) -> str:
        """
        Generate a sentence using the dynamic language engine.
        NO external LLM required!
        
        Examples:
            model.speak("القط", "يأكل", "السمك")  → "القط يأكل السمك"
            model.speak("الجو", "جميل")           → "الجو جميل"
        """
        if not self.language:
            return f"{subject} {action or ''} {obj or ''}".strip()
        
        return self.language.generate_simple(subject, action, obj)
    
    def describe(self, entity: str) -> str:
        """
        Generate a description of an entity from knowledge.
        Uses language engine + knowledge cells.
        """
        if not self.language:
            return f"لا أستطيع وصف {entity}"
        
        # Find the entity in our cells
        cell = self._find_cell_by_meta_value(entity)
        if not cell:
            return f"لا أعرف {entity}"
        
        # Generate description using language engine
        return self.language.generate_from_knowledge(cell, include_facts=True)
    
    def learn_language(self, text: str) -> int:
        """
        Learn vocabulary and patterns from text.
        Returns number of new words learned.
        
        Example:
            model.learn_language("القط الأسود يأكل السمك الطازج")
        """
        if not self.language:
            return 0
        
        before_count = len(self.language.vocab)
        self.language.vocab.learn_from_text(text)
        new_words = len(self.language.vocab) - before_count
        
        print(f"[LANGUAGE] Learned {new_words} new words from text")
        return new_words
    
    def persist_language(self) -> bool:
        """Save vocabulary and patterns to files."""
        if not self.language:
            return False
        
        self.language.vocab.persist()
        self.language.grammar.persist()
        print(f"[LANGUAGE] Saved {len(self.language.vocab)} words, {len(self.language.grammar)} patterns")
        return True

