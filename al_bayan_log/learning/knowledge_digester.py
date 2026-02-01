# -*- coding: utf-8 -*-
"""
Knowledge Digester 🍎
Transforms natural language lessons into executable logic.
"""

import json
import re

class KnowledgeDigester:
    def __init__(self, output_model):
        """
        :param output_model: Instance of SelfWritingModel to save the logic.
        """
        self.model = output_model

    def digest_lesson(self, topic: str, lesson_text: str):
        """
        Takes a lesson and converts it into a Logic Concept.
        Includes a Self-Correction Loop (IDE Simulation).
        """
        print(f"🍎 Digesting Lesson on: {topic}...")
        
        # 1. Structural Prompt for the Donor Brain
        base_system_prompt = """You are an AI Architect. Your goal is to convert a user's lesson into executable Python code (a Class or Functions).
        
        Output format MUST be a valid JSON block inside ```json ... ``` tags:
        {
            "concept_name": "NameOfTheConcept",
            "description": "Short summary",
            "methods": {
                "method_name": "def method_name(...): ... full code ..."
            }
        }
        
        Rules:
        - The code must be valid Python.
        - Include type hints.
        
        HARDWARE/PHYSICS API AVALIABLE:
        If the lesson involves physics (mass, distance, force), you MUST use the provided `core.physics_lib`.
        Example Usage:
        ```python
        from core.physics_lib import Kg, Meter, Force, Acceleration
        
        def calculate_force(m_val, a_val):
            # Inputs are floats, output is PhysicalQuantity
            mass = Kg(m_val) 
            acc = Acceleration(a_val)
            return mass * acc # Returns Force in Newtons
        ```
        """
        
        current_prompt = f"Topic: {topic}\nLesson: {lesson_text}\nConvert this to Python logic."
        
        max_retries = 3
        
        for attempt in range(max_retries):
            # 2. Ask Donor Brain
            response = self.model.ask_donor_brain(
                system_prompt=base_system_prompt,
                user_prompt=current_prompt,
                expect_code=False
            )
            
            if not response:
                return "❌ Failed to get response from Donor Brain."
                
            # 3. Parse JSON
            try:
                # Cleanup potential markdown wrapping
                cleaned_response = response
                if "```json" in response:
                     parts = response.split("```json")
                     if len(parts) > 1:
                         cleaned_response = parts[1].split("```")[0].strip()
                elif "```" in response:
                     cleaned_response = response.split("```")[1].strip()
                
                data = json.loads(cleaned_response)
                
                concept_name = data.get("concept_name", topic)
                methods = data.get("methods", {})
                
                if not methods:
                     return "⚠️ Donor Brain understood but produced no executable methods."
                
                # 4. 🛡️ IDE CHECK (The Self-Correction)
                # We assemble the code to test it
                full_test_code = f"class {concept_name}:\n    pass\n\n"
                for m_name, m_code in methods.items():
                    full_test_code += m_code + "\n\n"
                    
                print(f"   🧪 Testing Code (Attempt {attempt+1}/{max_retries})...", end="\r")
                test_result = self.model.immune_system.sandbox_test(full_test_code)
                
                if not test_result["valid"]:
                    # FAILED! Loop back.
                    error_msg = test_result.get("traceback", test_result.get("error"))
                    print(f"   ❌ IDE Error: {test_result['error']}")
                    print(f"   🔄 Requesting Fix from Donor Brain...")
                    
                    current_prompt = f"""
                    Your previous attempt to write '{concept_name}' failed to compile/run.
                    
                    ERROR:
                    {error_msg}
                    
                    PREVIOUS CODE:
                    {full_test_code}
                    
                    Please FIX the code and return the JSON again.
                    """
                    continue # Retry loop
                
                # SUCCESS!
                print(f"   ✅ Tests Passed! Saving '{concept_name}'...")
                
                # 5. Save to Memory
                cell = self.model.learn_logic_concept(concept_name, methods)
                
                if cell:
                    return f"✅ Lesson Learned! Concept '{concept_name}' is now active logic."
                else:
                    return "❌ Failed to save logic (Immune System rejection?)."
                    
            except json.JSONDecodeError:
                print(f"⚠️ JSON Parse Error. Retrying...")
                current_prompt += "\n\nERROR: The output was not valid JSON. Please format strictly as JSON."
                
            except Exception as e:
                return f"❌ Digestion Error: {e}"
                
        return "❌ Failed to generate valid code after multiple attempts."
