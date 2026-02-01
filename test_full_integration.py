#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test: Full Al-Bayan Integration"""

import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing Al-Bayan Full Integration")
print("=" * 60)

# 1. Test Arabic NLP
print("\n[1] Testing Arabic NLP...")
from al_bayan_log.nlp import segment, pos_tag, extract_entities

text = "المهندس أحمد يسكن في الرياض"
segments = segment(text)
print(f"   Text: {text}")
print(f"   Segments: {len(segments)} words")
for s in segments:
    print(f"      - {s['original']} -> stem: {s['stem']}, prefix: {s['prefix']}, suffix: {s['suffix']}")

entities = extract_entities(text)
print(f"   Entities: {entities}")

# 2. Test NeuroSymbolicBridge
print("\n[2] Testing NeuroSymbolicBridge...")
from al_bayan_log.inference import NeuroSymbolicBridge, Concept, Fact

bridge = NeuroSymbolicBridge()
bridge.learn_concept(Concept(
    name="human",
    properties=["thinks", "speaks"],
    instances=["Socrates", "Plato"]
))
bridge.learn_fact(Fact("Socrates", "is_a", "human"))

# Test deduction
results = bridge.deduce("human", "")
print(f"   Deductions: {len(results)} found")
for r in results:
    print(f"      - {r.conclusion} (confidence: {r.confidence})")

# 3. Test Logical Engine
print("\n[3] Testing LogicalEngine...")
from al_bayan_log.compiler import LogicalEngine, Term, Predicate, Fact as LogicalFact

engine = LogicalEngine()
engine.add_fact(LogicalFact(Predicate("mortal", [Term("all_humans")])))
print(f"   Facts in engine: {len(engine.knowledge_base)} predicates")

# 4. Test ChatEngine with NLP
print("\n[4] Testing ChatEngine with NLP...")
from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine

model = SelfWritingModel(os.path.dirname(os.path.abspath(__file__)))
chat = ChatEngine(model)

analysis = chat.analyze_text(text)
print(f"   Analysis available: {type(analysis)}")
print(f"   Keys: {list(analysis.keys())}")

print("\n" + "=" * 60)
print("[SUCCESS] All integrations working!")
print("=" * 60)
