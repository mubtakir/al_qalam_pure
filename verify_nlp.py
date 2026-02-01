import os
import sys
import io

# Ensure UTF-8 output for Arabic on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine
from core.linguistic_narrator import LinguisticNarrator

def verify_nlp():
    print("--- Verification: Al-Qalam Natural NLP ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    chat = ChatEngine(model)
    narrator = LinguisticNarrator()

    # 1. Test Synonym Handling
    print("[1] Testing Synonym: 'خالد يقطن في الفضاء'")
    resp = chat.process("خالد يقطن في الفضاء")
    print(f"Response: {resp}")
    if "تم تسجيل حقيقة" in resp:
        print("[Success] Corrected mapped 'يقطن' to 'Fact' intent.")

    # 2. Test Rule Narration
    print("\n[2] Testing Eloquent Rule Narration:")
    rule = "rule_induct_concept_human_item_apple"
    eloquent = narrator.narrate_inference(rule)
    print(f"Technical: {rule}")
    print(f"Eloquent: {eloquent}")

    # 3. Test Thought Narration (Analogy)
    print("\n[3] Testing Thought Narration (Analogy):")
    thought = "قياس منطقي: بما أن Human يرتبط بـ Apple، هل تفكر أن Engineer قد يرتبط به أيضاً؟"
    eloquent = narrator.narrate_thought(thought)
    print(f"Technical: {thought}")
    print(f"Eloquent: {eloquent}")

    print("\n--- NLP Verification Complete ---")

if __name__ == "__main__":
    verify_nlp()
