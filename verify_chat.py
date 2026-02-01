import os
import sys
import io

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine

def verify_chat():
    print("--- Verification: Al-Qalam Chat Engine ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    chat = ChatEngine(model)
    
    # 1. Test Learning intent
    print("[1] Testing: 'تعلم أن الأطباء هم كمال وسامي'")
    resp = chat.process("تعلم أن الأطباء هم كمال وسامي")
    print(f"Response: {resp}")
    if "تم تعلم مفهوم 'الأطباء'" in resp:
        print("[Success] Identified 'Learn' intent.")
    
    # 2. Test Fact intent
    print("[2] Testing: 'كمال يعمل في المستشفى'")
    # Note: item_hospital needs to exist
    from core.dynamic_cell import DynamicCell
    if "item_hospital" not in model.cells:
        model.cells["item_hospital"] = DynamicCell("item_hospital", "item", 0.0, {"name": "المستشفى"})
        model.persist_cells()
        
    resp = chat.process("كمال يعمل في المستشفى")
    print(f"Response: {resp}")
    if "تم تسجيل حقيقة" in resp:
        print("[Success] Identified 'Fact' intent.")
        
    # 3. Test Induction intent
    print("[3] Testing: 'استنتج القواعد'")
    resp = chat.process("استنتج")
    print(f"Response: {resp}")
    
    # 4. Test Query intent
    print("[4] Testing: 'ماذا تعرف عن كمال?'")
    resp = chat.process("ماذا تعرف عن كمال?")
    print(f"Response: {resp}")
    if "معلومات عن كمال" in resp:
        print("[Success] Identified 'Query' intent.")

    print("\n--- Chat Verification Complete ---")

if __name__ == "__main__":
    verify_chat()
