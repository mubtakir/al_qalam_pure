#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational Interface for Al-Qalam Pure.
"""

import sys
import os

# Ensure the current directory is in sys.path
sys.path.insert(0, os.getcwd())

from core.self_writing_model import SelfWritingModel
from core.chat_engine import ChatEngine

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = SelfWritingModel(base_dir)
    chat = ChatEngine(model)
    
    print("\n" + "="*50)
    print("✨ محادثة Al-Qalam Pure: الذكاء الرمزي الحي ✨")
    print("="*50)
    print("اكتب 'خروج' للإنهاء.\n")
    
    while True:
        try:
            user_input = input("👤 أنت: ").strip()
            
            if user_input.lower() in ["خروج", "exit", "quit"]:
                print("\nوداعاً! تم حفظ كافة التعديلات في الكود المصدري.")
                break
            
            if not user_input:
                continue
                
            response = chat.process(user_input)
            print(f"🤖 القلم: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"⚠️ خطأ: {e}")

if __name__ == "__main__":
    main()
