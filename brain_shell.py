#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import io

# Fix for Windows terminal encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine
from core.cortex import Cortex

def brain_shell():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    cortex = Cortex(model)
    chat = ChatEngine(model)
    
    print("\n" + "🧠"*10 + " AL-QALAM CENTRAL BRAIN (CORTEX) " + "🧠"*10)
    print("نظام التفكير المبادر - اكتب 'خروج' للإنهاء.\n")
    
    while True:
        try:
            # 1. Run Cortex background thinking
            thoughts = cortex.think()
            
            # Show current dream if stable
            dream = cortex.dreamer.current_dream
            if dream:
                print(f"\n[💤 حالة الحلم الجارية]: {dream}")

            if thoughts:
                print(f"[خواطر العقل المركزي]:")
                for t in thoughts:
                    # Use narrator for all thoughts
                    eloquent_thought = chat.narrator.narrate_thought(t)
                    print(f" 💭 {eloquent_thought}")
                
                 # Display Liquid Voice (Contextual)
                try:
                    liquid_status = cortex.express_state()
                    mood_icon = "🟢" if cortex.stress_level < 0 else "🔴"
                    print(f" {mood_icon} [Liquid Voice]: {liquid_status}")
                except Exception:
                     pass

                print("-" * 30)

            # 2. Get User Input
            user_input = input("\n👤 أنت: ").strip()
            
            if user_input.lower() in ["خروج", "exit", "quit"]:
                break
            
            if not user_input:
                continue

            # Debug: Manual Stress Control
            if user_input.startswith("/stress"):
                try:
                    val = float(user_input.split()[1])
                    cortex.stress_level = val
                    print(f"⚙️ [DEBUG] Stress Level set to: {val}")
                except:
                    print("Usage: /stress <float_value>")
                continue
                
            # 3. Process conversation
            response = chat.process(user_input)
            print(f"🤖 القلم: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ خطأ: {e}")

if __name__ == "__main__":
    brain_shell()
