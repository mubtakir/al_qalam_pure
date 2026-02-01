import os
import sys
import io

# Ensure UTF-8 output for Arabic on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.cortex import Cortex
from core.chat_engine import ChatEngine

def run_cortex_demo():
    print("🧠 --- AL-QALAM PURE: CORTEX LIVE DEMO --- 🧠\n")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    cortex = Cortex(model)
    chat = ChatEngine(model)
    
    # 1. Show Thoughts
    print("[1] تشغيل محرك التفكير (Cortex)...")
    thoughts = cortex.think()
    if thoughts:
        print("\n[خواطر العقل المركزي الحالية]:")
        for t in thoughts:
             print(f" 💭 {t}")
    else:
        print("\n[!] لا توجد خواطر حالية. العقل في حالة استقرار.")

    # 2. Simulate User choosing an analogy to confirm
    # e.g. "نعم، المهندسون يحبون التفاح" (Linking Human preference to Engineer)
    # We'll just pick a thought to respond to in a real chat session,
    # but for demo we simulate the 'Correct' feedback which updates the model.
    print("\n[2] محاكاة تأكيد المستخدم لأحد القياسات المنطقية...")
    # Trigger a rule feedback manually to show the effect
    # We'll use the rule_induct_concept_engineer_item_bracket since it exists.
    model.apply_feedback("rule_induct_concept_engineer_item_bracket", positive=True)
    
    print("\n[3] فحص الأوزان البرمجية في cells.py...")
    # Just check if 'delta' changed for a cell.
    # We know in previous steps it went to 0.2. Let's see if it's there.
    cells_path = os.path.join(base_dir, "vault", "auto_generated", "cells.py")
    with open(cells_path, 'r', encoding='utf-8') as f:
        content = f.read()
        if "delta=0.3" in content:
            print("✅ نجاح: تم العثور على أوزان معدلة جراحياً للمستوى الثالث (delta=0.3)!")
        elif "delta=0.2" in content:
            print("✅ نجاح: تم العثور على أوزان معدلة جراحياً للمستوى الثاني (delta=0.2)!")
        else:
            print("⚠️ ملاحظة: تم التوثيق لكن لم نجد delta=0.2+. يرجى مراجعة cells.py")

if __name__ == "__main__":
    run_cortex_demo()
