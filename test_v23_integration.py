#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Al-Qalam V2.3 Integration Test
Tests the full loop: LLM → ImmuneSystem → Learning
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from core.self_writing_model import SelfWritingModel, IMMUNE_AVAILABLE, LLM_AVAILABLE

def test_full_integration():
    print("=" * 60)
    print("🧪 Al-Qalam V2.3 Integration Test")
    print("=" * 60)
    
    # 1. Create Model
    print("\n📦 Creating SelfWritingModel...")
    model = SelfWritingModel(".")
    
    # 2. Check Components
    print("\n🔧 Component Status:")
    print(f"   ImmuneSystem: {'✅ Active' if model.immune else '❌ Missing'}")
    print(f"   LogicalBridge: {'✅ Active' if model.logic else '❌ Missing'}")
    print(f"   LLM Available: {'✅ Yes' if LLM_AVAILABLE else '⚠️ No (will skip LLM tests)'}")
    
    # 3. Test ImmuneSystem
    print("\n🛡️ Testing ImmuneSystem...")
    
    # Valid code
    result = model.safe_generate_code("x = 1 + 1", "simple math")
    print(f"   Valid code test: {'✅ Passed' if result['valid'] else '❌ Failed'}")
    
    # Invalid code
    result = model.safe_generate_code("x = 1 +", "broken code")
    print(f"   Invalid code rejected: {'✅ Passed' if not result['valid'] else '❌ Failed'}")
    
    # 4. Test learn_logic_concept
    print("\n📚 Testing learn_logic_concept...")
    
    test_code = '''def greet(name: str) -> str:
    return f"Hello, {name}!"
'''
    
    cell = model.learn_logic_concept("Greeting", {"greet": test_code})
    if cell:
        print(f"   ✅ Logic concept learned: {cell.id}")
        print(f"   Methods: {cell.metadata.get('methods', [])}")
    else:
        print("   ❌ Failed to learn logic concept")
    
    # 5. Test LLM (if available)
    if LLM_AVAILABLE:
        print("\n🧠 Testing LLM...")
        try:
            response = model.ask_llm("What is 2 + 2?")
            if response and "[LLM" not in response:
                print(f"   ✅ LLM responded: {response[:50]}...")
            else:
                print(f"   ⚠️ LLM not loaded or error")
        except Exception as e:
            print(f"   ⚠️ LLM test skipped: {e}")
    else:
        print("\n⚠️ LLM tests skipped (not available)")
    
    # 6. Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"   Total cells: {len(model.cells)}")
    print(f"   Total rules: {len(model.rules)}")
    print(f"   ImmuneSystem: {'Active' if model.immune else 'Inactive'}")
    print(f"   LLM Bridge: {'Available' if LLM_AVAILABLE else 'Not available'}")
    print("\n✅ Integration test complete!")

if __name__ == "__main__":
    test_full_integration()
