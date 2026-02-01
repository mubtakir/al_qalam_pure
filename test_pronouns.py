import os
import sys
import io

# Ensure UTF-8 output for Arabic on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine

def test_pronouns():
    print("--- Test: Pronoun Resolution ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    chat = ChatEngine(model)
    
    # 1. Establish context
    print("[1] Input: 'خالد يشتغل في الفضاء'")
    chat.process("خالد يشتغل في الفضاء")
    print(f"Current last_subject: {chat.last_subject}")
    
    # 2. Use pronoun
    print("\n[2] Input: 'هو يحب التفاح'")
    resp = chat.process("هو يحب التفاح")
    print(f"Response: {resp}")
    
    if "خالد" in resp and "يحب" in resp:
         print("\n✅ Success: Pronoun 'هو' resolved to 'خالد'!")
    else:
         print("\n❌ Failure: Pronoun resolution failed.")

if __name__ == "__main__":
    test_pronouns()
