import os
import sys
import io

# Ensure UTF-8 output for Arabic on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine
from core.dynamic_cell import DynamicCell

def run_demo():
    print("🚀 --- AL-QALAM PURE: LIVE DEMO --- 🚀\n")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    chat = ChatEngine(model)
    
    # Pre-check: ensure 'Space' item exists for the demo facts
    if "item_space" not in model.cells:
        model.cells["item_space"] = DynamicCell("item_space", "item", 0.0, {"name": "الفضاء"})
        model.persist_cells()

    steps = [
        "تعلم أن الرواد هم نيل و باز",
        "نيل يسكن في الفضاء",
        "باز يسكن في الفضاء",
        "استنتج",
        "صحيح" # Feedback for the induced rule: 'الرواد' -> 'الفضاء'
    ]
    
    for i, input_text in enumerate(steps, 1):
        print(f"👤 خطوة {i} (المستخدم): {input_text}")
        response = chat.process(input_text)
        print(f"🤖 القلم: {response}\n")

    print("🔍 --- فحص الكود المصدري بعد المحادثة ---")
    cells_path = os.path.join(base_dir, "vault", "auto_generated", "cells.py")
    with open(cells_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Find connection logic for "الرواد"
        if "inst_الرواد" in content and "delta=+0.10" in content:
            print("✅ نجاح: تم العثور على أوزان برمجية معدلة جراحياً (delta=+0.10)!")
        else:
             print("⚠️ ملاحظة: تأكد من أن القاعدة استنتجت بنجاح في cells.py")

if __name__ == "__main__":
    run_demo()
