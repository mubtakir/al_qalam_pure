#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: Bayan Logical Engine Integration
"""

import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.self_writing_model import SelfWritingModel

def test_integration():
    print("=" * 50)
    print("Test: Bayan Logical Engine Integration")
    print("=" * 50)
    
    # 1. Create model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    
    print("\n[OK] 1. Model created with LogicalBridge")
    print(f"   - Logic Engine Type: {type(model.logic).__name__}")
    
    # 2. Learn a concept
    print("\n[LEARN] 2. Learning new concept...")
    model.learn_concept("fruit", ["apple", "orange", "banana"])
    
    # 3. Check logical facts
    facts = model.logic.get_all_facts()
    print(f"\n[FACTS] 3. Logical facts ({len(facts)} total):")
    for fact in facts[:5]:  # Show first 5
        print(f"   - {fact}")
    if len(facts) > 5:
        print(f"   ... and {len(facts) - 5} more facts")
    
    # 4. Query via logical engine
    print("\n[QUERY] 4. Logical query:")
    results = model.logic.query("type", ["concept_fruit", "?Type"])
    print(f"   Type of 'concept_fruit': {results}")
    
    # 5. Query connections
    print("\n[QUERY] 5. Connection query:")
    results = model.logic.query("connected", ["concept_fruit", "?Target", "?Weight"])
    print(f"   Connections from concept_fruit: {len(results)} found")
    for r in results[:3]:
        print(f"   - {r}")
    
    print("\n" + "=" * 50)
    print("[SUCCESS] Integration test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_integration()
