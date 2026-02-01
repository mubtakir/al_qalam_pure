#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogicalBridge: Integration layer between Al-Qalam cells and Bayan's LogicalEngine.
جسر منطقي: طبقة تكامل بين خلايا القلم ومحرك البيان المنطقي
"""

from typing import Dict, List, Optional, Any
from al_bayan_log.compiler.logical_engine import LogicalEngine, Term, Predicate, Fact, Rule


class LogicalBridge:
    """Bridges DynamicCells with Bayan's Prolog-like LogicalEngine."""
    
    def __init__(self):
        self.engine = LogicalEngine()
        self._synced_cells: set = set()
    
    def sync_cell(self, cell) -> None:
        """Converts a DynamicCell into logical facts."""
        if cell.id in self._synced_cells:
            return
            
        # Fact 1: Cell type
        # e.g., type(concept_fruit, concept).
        self.engine.add_fact(Fact(Predicate("type", [
            Term(cell.id),
            Term(cell.type)
        ])))
        
        # Fact 2: Cell metadata (name/value)
        if "name" in cell.metadata:
            self.engine.add_fact(Fact(Predicate("name", [
                Term(cell.id),
                Term(cell.metadata["name"])
            ])))
        
        if "value" in cell.metadata:
            self.engine.add_fact(Fact(Predicate("value", [
                Term(cell.id),
                Term(cell.metadata["value"])
            ])))
        
        # Fact 3: Connections as relations
        # e.g., connected(inst_apple_0, concept_fruit, 0.9).
        for target_id, param in cell.connections.items():
            weight = param.value if hasattr(param, 'value') else param
            self.engine.add_fact(Fact(Predicate("connected", [
                Term(cell.id),
                Term(target_id),
                Term(str(weight))
            ])))
        
        self._synced_cells.add(cell.id)
    
    def sync_all_cells(self, cells: Dict) -> None:
        """Syncs all cells from the model."""
        for cell in cells.values():
            self.sync_cell(cell)
    
    def add_rule(self, head_name: str, head_args: List[str], 
                 body: List[tuple]) -> None:
        """
        Adds a logical rule.
        Example: add_rule("mortal", ["?X"], [("human", ["?X"])])
        Creates: mortal(?X) :- human(?X).
        """
        head_terms = [Term(a[1:], is_variable=True) if a.startswith("?") else Term(a) 
                      for a in head_args]
        head = Predicate(head_name, head_terms)
        
        body_preds = []
        for pred_name, pred_args in body:
            terms = [Term(a[1:], is_variable=True) if a.startswith("?") else Term(a) 
                     for a in pred_args]
            body_preds.append(Predicate(pred_name, terms))
        
        self.engine.add_rule(Rule(head, body_preds))
    
    def query(self, pred_name: str, args: List[str]) -> List[Dict]:
        """
        Executes a logical query.
        Example: query("connected", ["inst_apple_0", "?Target"])
        Returns: [{"Target": "concept_fruit"}, ...]
        """
        terms = [Term(a[1:], is_variable=True) if a.startswith("?") else Term(a) 
                 for a in args]
        goal = Predicate(pred_name, terms)
        
        results = self.engine.query(goal)
        
        # Convert Substitutions to readable dicts
        output = []
        for sub in results:
            bindings = {}
            for var, val in sub.bindings.items():
                if isinstance(val, Term):
                    bindings[var] = val.value
                else:
                    bindings[var] = val
            output.append(bindings)
        
        return output
    
    def query_natural(self, question: str) -> Optional[str]:
        """
        Experimental: Parses a natural Arabic question into a query.
        Example: "ما نوع التفاحة؟" -> query("type", ["inst_apple_0", "?Type"])
        """
        # Simple pattern matching for now
        if "نوع" in question or "type" in question.lower():
            # Extract entity name (simplified)
            return "استعلام عن النوع - يحتاج لتحليل NLP أعمق"
        
        return None
    
    def get_all_facts(self) -> List[str]:
        """Returns all facts as human-readable strings."""
        facts = []
        for pred_name, items in self.engine.knowledge_base.items():
            for item in items:
                if isinstance(item, Fact):
                    facts.append(str(item))
        return facts
    
    def clear(self) -> None:
        """Clears all logical knowledge."""
        self.engine = LogicalEngine()
        self._synced_cells.clear()
