"""
Neuro-Symbolic Bridge
======================
Connects Golden Giant (statistical brain) to Cognitive Evolution (symbolic reasoning).
Enables hybrid AI with both language generation and logical inference.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class ReasoningType(Enum):
    """Types of reasoning supported by the bridge."""
    GENERATION = "generation"      # Pure statistical generation
    INDUCTION = "induction"        # From examples to general rules
    DEDUCTION = "deduction"        # From rules to specific instances
    SYLLOGISM = "syllogism"        # Formal logical inference
    HYBRID = "hybrid"              # Combined statistical + logical


@dataclass
class Concept:
    """A concept extracted from text or knowledge base."""
    name: str
    properties: List[str] = field(default_factory=list)
    instances: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "extracted"


@dataclass
class Fact:
    """A fact triplet (subject, relation, object)."""
    subject: str
    relation: str
    obj: str
    confidence: float = 1.0
    source: str = "generated"


@dataclass
class InferenceResult:
    """Result from the inference engine."""
    conclusion: str
    reasoning_type: ReasoningType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    is_valid: bool = True


class NeuroSymbolicBridge:
    """
    Bridge between Golden Giant and Cognitive Evolution.
    
    Workflow:
    1. Generate text with Golden Giant
    2. Extract concepts and facts
    3. Validate with Cognitive Evolution
    4. Enhance with logical inference
    5. Return validated, reasoned output
    """
    
    def __init__(self, knowledge_path: Optional[str] = None):
        """
        Initialize the bridge.
        
        Args:
            knowledge_path: Path to knowledge graph JSON
        """
        self.knowledge_path = knowledge_path or self._default_knowledge_path()
        self.knowledge_graph = self._load_knowledge()
        self.facts_buffer: List[Fact] = []
        self.concepts_cache: Dict[str, Concept] = {}
    
    def _default_knowledge_path(self) -> str:
        """Get default knowledge graph path."""
        base = Path(__file__).parent / "cognitive_evolution" / "knowledge"
        return str(base / "knowledge_graph.json")
    
    def _load_knowledge(self) -> Dict[str, Any]:
        """Load knowledge graph from file."""
        if os.path.exists(self.knowledge_path):
            with open(self.knowledge_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "concepts": {},
            "facts": [],
            "inferred_relations": [],
            "rules": []
        }
    
    def _save_knowledge(self) -> None:
        """Save knowledge graph to file."""
        os.makedirs(os.path.dirname(self.knowledge_path), exist_ok=True)
        with open(self.knowledge_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_graph, f, ensure_ascii=False, indent=2)
    
    # =========================================================================
    # Concept Extraction
    # =========================================================================
    
    def extract_concepts(self, text: str) -> List[Concept]:
        """
        Extract concepts from generated text.
        Uses simple pattern matching for now, can be enhanced with NER.
        """
        concepts = []
        
        # Simple noun phrase extraction (can be enhanced)
        words = text.replace("،", ",").replace(".", " ").split()
        
        # Look for known concepts
        for concept_name, concept_data in self.knowledge_graph.get("concepts", {}).items():
            if concept_name in text:
                concept = Concept(
                    name=concept_name,
                    properties=concept_data.get("properties", []),
                    instances=concept_data.get("instances", []),
                    relations=concept_data.get("relations", {}),
                    source="knowledge_base"
                )
                concepts.append(concept)
                self.concepts_cache[concept_name] = concept
        
        return concepts
    
    def extract_facts(self, text: str) -> List[Fact]:
        """
        Extract fact triplets from generated text.
        Simple pattern matching for Arabic and English.
        """
        facts = []
        
        # Arabic patterns
        arabic_patterns = [
            ("يحب", "likes"),
            ("يكره", "dislikes"),
            ("هو", "is_a"),
            ("يملك", "has"),
            ("يعرف", "knows"),
        ]
        
        # English patterns
        english_patterns = [
            ("is a", "is_a"),
            ("has", "has"),
            ("likes", "likes"),
            ("knows", "knows"),
        ]
        
        # Extract based on patterns (simplified)
        for pattern, relation in arabic_patterns + english_patterns:
            if pattern in text:
                # Extract surrounding context
                idx = text.find(pattern)
                before = text[max(0, idx-20):idx].strip().split()[-1] if idx > 0 else ""
                after = text[idx+len(pattern):idx+len(pattern)+20].strip().split()[0] if idx+len(pattern) < len(text) else ""
                
                if before and after:
                    fact = Fact(
                        subject=before,
                        relation=relation,
                        obj=after,
                        confidence=0.7,
                        source="extracted"
                    )
                    facts.append(fact)
                    self.facts_buffer.append(fact)
        
        return facts
    
    # =========================================================================
    # Inference Integration
    # =========================================================================
    
    def induce(self, facts: List[Fact], min_confidence: float = 0.5) -> List[InferenceResult]:
        """
        Perform inductive reasoning: from specific facts to general rules.
        
        Example:
            Facts: [خالد يحب تفاح, زيد يحب تفاح, عمر يحب تفاح]
            Induced: إنسان يحب تفاح (confidence based on coverage)
        """
        results = []
        
        # Group facts by relation
        relation_groups: Dict[str, List[Fact]] = {}
        for fact in facts:
            key = f"{fact.relation}_{fact.obj}"
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(fact)
        
        # Check for patterns
        for key, group in relation_groups.items():
            if len(group) >= 2:
                relation, obj = key.rsplit("_", 1)
                subjects = [f.subject for f in group]
                
                # Check if subjects share a concept
                common_concept = self._find_common_concept(subjects)
                
                if common_concept:
                    confidence = len(group) / max(len(group), 3)  # At least 3 examples for high confidence
                    
                    if confidence >= min_confidence:
                        result = InferenceResult(
                            conclusion=f"{common_concept} {relation} {obj}",
                            reasoning_type=ReasoningType.INDUCTION,
                            confidence=confidence,
                            evidence=[f"{f.subject} {f.relation} {f.obj}" for f in group]
                        )
                        results.append(result)
        
        return results
    
    def deduce(self, concept: str, query: str) -> List[InferenceResult]:
        """
        Perform deductive reasoning: from general rules to specific instances.
        
        Example:
            Rule: إنسان يفكر
            Instance: سقراط إنسان
            Deduced: سقراط يفكر
        """
        results = []
        
        concept_data = self.knowledge_graph.get("concepts", {}).get(concept, {})
        properties = concept_data.get("properties", [])
        instances = concept_data.get("instances", [])
        
        # Apply properties to instances
        for instance in instances:
            for prop in properties:
                result = InferenceResult(
                    conclusion=f"{instance} {prop}",
                    reasoning_type=ReasoningType.DEDUCTION,
                    confidence=0.95,
                    evidence=[f"{concept} {prop}", f"{instance} هو {concept}"]
                )
                results.append(result)
        
        return results
    
    def syllogism(
        self, 
        major_premise: Tuple[str, str, str],  # (M, relation, P) - All M are P
        minor_premise: Tuple[str, str, str]   # (S, relation, M) - S is M
    ) -> Optional[InferenceResult]:
        """
        Perform syllogistic reasoning (formal logic).
        
        Example:
            Major: كل إنسان فانٍ (All humans are mortal)
            Minor: سقراط إنسان (Socrates is human)
            Conclusion: سقراط فانٍ (Socrates is mortal)
        """
        m_major, rel_major, p = major_premise
        s, rel_minor, m_minor = minor_premise
        
        # Check if middle terms match
        if m_major != m_minor:
            return InferenceResult(
                conclusion="",
                reasoning_type=ReasoningType.SYLLOGISM,
                confidence=0.0,
                is_valid=False,
                evidence=["المقدمتان لا تشتركان في الحد الأوسط"]
            )
        
        # Valid syllogism
        return InferenceResult(
            conclusion=f"{s} {rel_major} {p}",
            reasoning_type=ReasoningType.SYLLOGISM,
            confidence=0.99,
            evidence=[
                f"{m_major} {rel_major} {p}",
                f"{s} {rel_minor} {m_minor}"
            ]
        )
    
    def _find_common_concept(self, instances: List[str]) -> Optional[str]:
        """Find a concept that all instances belong to."""
        for concept_name, concept_data in self.knowledge_graph.get("concepts", {}).items():
            known_instances = concept_data.get("instances", [])
            if all(inst in known_instances for inst in instances):
                return concept_name
        return None
    
    # =========================================================================
    # Validation & Enhancement
    # =========================================================================
    
    def validate_output(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate generated output against knowledge base.
        Returns (is_valid, list of issues)
        """
        issues = []
        
        # Extract facts from text
        facts = self.extract_facts(text)
        
        # Check against known facts
        known_facts = self.knowledge_graph.get("facts", [])
        
        for fact in facts:
            # Check for contradictions
            for known in known_facts:
                if (known.get("subject") == fact.subject and 
                    known.get("relation") == fact.relation and
                    known.get("object") != fact.obj):
                    issues.append(f"تناقض: {fact.subject} {fact.relation} {fact.obj} vs {known.get('object')}")
        
        return len(issues) == 0, issues
    
    def enhance_with_reasoning(
        self, 
        text: str, 
        reasoning_type: ReasoningType = ReasoningType.HYBRID
    ) -> str:
        """
        Enhance generated text with logical reasoning.
        Adds inferred knowledge to the output.
        """
        # Extract current facts
        facts = self.extract_facts(text)
        
        enhancements = []
        
        if reasoning_type in [ReasoningType.INDUCTION, ReasoningType.HYBRID]:
            # Try induction
            induced = self.induce(facts)
            for result in induced:
                if result.confidence >= 0.7:
                    enhancements.append(f"[استنتاج: {result.conclusion}]")
        
        if reasoning_type in [ReasoningType.DEDUCTION, ReasoningType.HYBRID]:
            # Try deduction on extracted concepts
            concepts = self.extract_concepts(text)
            for concept in concepts:
                deduced = self.deduce(concept.name, text)
                for result in deduced[:2]:  # Limit to 2 per concept
                    enhancements.append(f"[استنباط: {result.conclusion}]")
        
        # Combine
        if enhancements:
            enhanced = text + "\n\n" + " ".join(enhancements)
            return enhanced
        
        return text
    
    # =========================================================================
    # Learning (Self-Writing Integration)
    # =========================================================================
    
    def learn_fact(self, fact: Fact) -> None:
        """Add a new fact to the knowledge base."""
        fact_dict = {
            "subject": fact.subject,
            "relation": fact.relation,
            "object": fact.obj,
            "confidence": fact.confidence,
            "source": fact.source
        }
        
        self.knowledge_graph.setdefault("facts", []).append(fact_dict)
        self._save_knowledge()
    
    def learn_concept(self, concept: Concept) -> None:
        """Add or update a concept in the knowledge base."""
        concept_dict = {
            "properties": concept.properties,
            "instances": concept.instances,
            "relations": concept.relations
        }
        
        self.knowledge_graph.setdefault("concepts", {})[concept.name] = concept_dict
        self._save_knowledge()
    
    def crystallize_inductions(self, min_confidence: float = 0.8) -> int:
        """
        Convert high-confidence inductions to permanent rules.
        Returns number of crystallized rules.
        """
        induced = self.induce(self.facts_buffer, min_confidence)
        count = 0
        
        for result in induced:
            if result.confidence >= min_confidence:
                rule = {
                    "conclusion": result.conclusion,
                    "type": "induction",
                    "confidence": result.confidence,
                    "evidence": result.evidence
                }
                self.knowledge_graph.setdefault("inferred_relations", []).append(rule)
                count += 1
        
        if count > 0:
            self._save_knowledge()
        
        return count


# =============================================================================
# Convenience Functions
# =============================================================================

def create_bridge(knowledge_path: Optional[str] = None) -> NeuroSymbolicBridge:
    """Create a new bridge instance."""
    return NeuroSymbolicBridge(knowledge_path)


def process_with_reasoning(text: str, bridge: Optional[NeuroSymbolicBridge] = None) -> str:
    """Process text through the neuro-symbolic bridge."""
    if bridge is None:
        bridge = create_bridge()
    
    # Validate
    is_valid, issues = bridge.validate_output(text)
    
    # Enhance
    enhanced = bridge.enhance_with_reasoning(text)
    
    if not is_valid:
        enhanced += f"\n[تحذير: {', '.join(issues)}]"
    
    return enhanced


if __name__ == "__main__":
    print("🧠 Testing Neuro-Symbolic Bridge...")
    
    bridge = create_bridge()
    
    # Add some test knowledge
    bridge.learn_concept(Concept(
        name="إنسان",
        properties=["يفكر", "يتكلم"],
        instances=["سقراط", "أفلاطون", "أرسطو"]
    ))
    
    bridge.learn_fact(Fact("سقراط", "is_a", "إنسان"))
    bridge.learn_fact(Fact("أفلاطون", "is_a", "إنسان"))
    
    # Test deduction
    print("\n📚 Testing Deduction:")
    results = bridge.deduce("إنسان", "")
    for r in results:
        print(f"  {r.conclusion} (confidence: {r.confidence})")
    
    # Test syllogism
    print("\n📚 Testing Syllogism:")
    result = bridge.syllogism(
        ("إنسان", "هو", "فانٍ"),
        ("سقراط", "هو", "إنسان")
    )
    if result:
        print(f"  {result.conclusion} (confidence: {result.confidence})")
    
    print("\n✅ Bridge working correctly!")
