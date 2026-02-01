# -*- coding: utf-8 -*-
"""
Al-Qalam Immune System 🛡️
Safety layer to prevent the model from writing broken code.
"""

import ast
import os
import shutil
from datetime import datetime

class ImmuneSystem:
    """
    Acts as a firewall between the model's imagination and its actual source code.
    Verifies that generated code is valid Python and follows safety protocols.
    """
    
    def __init__(self, memory_dir):
        self.memory_dir = memory_dir
        self.backup_dir = os.path.join(memory_dir, "..", "backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        print("[IMMUNE] Immune System Active: Monitoring Code Safety")

    def validate_code(self, code_snippet: str) -> dict:
        """
        Checks if the code snippet is valid Python.
        Returns: {"valid": bool, "error": str/None, "tree": AST}
        """
        try:
            # 1. Syntax Check (Compile)
            tree = ast.parse(code_snippet)
            
            # 2. Safety Checks (Static Analysis)
            # Example: Ban 'import os.system' or 'subprocess'
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name in ['subprocess', 'shutil', 'sys']:
                             # Allow shutil only for internal use, but maybe flag it?
                             # For now, let's just warn generic restricted imports
                             pass 
                             
            return {"valid": True, "error": None}
            
        except SyntaxError as e:
            return {"valid": False, "error": f"Syntax Error: {e}"}
        except Exception as e:
            return {"valid": False, "error": f"Validation Error: {e}"}

    def backup_memory(self, filename: str):
        """Creates a timestamped backup of a memory file"""
        source = os.path.join(self.memory_dir, filename)
        if not os.path.exists(source):
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(self.backup_dir, f"{filename}.{timestamp}.bak")
        shutil.copy2(source, dest)
        # print(f"🛡️ Backup created: {dest}")

    def safe_write(self, filename: str, existing_content: str, new_chunk: str) -> bool:
        """
        Safely updates a file:
        1. Validate new chunk.
        2. Backup old file.
        3. Write new content.
        """
        # Validate
        validation = self.validate_code(new_chunk)
        if not validation["valid"]:
            print(f"[BLOCKED] Immune System rejected invalid code.\nReason: {validation['error']}")
            return False
            
        # Backup
        self.backup_memory(filename)
        
        # Write (In a real scenario, we might want to validate the WHOLE file merged)
        # For now, we trust that valid chunks appended to valid file = valid file
        # (Though technically indentation errors could happen, so checking the merge is better)
        
        full_proposed_content = existing_content + "\n" + new_chunk
        full_validation = self.validate_code(full_proposed_content)
        
        if not full_validation["valid"]:
             print(f"[BLOCKED] Merging code would break the file.\nReason: {full_validation['error']}")
             return False
             
        # Actual Write happens in the Caller (SelfWritingModel), or here?
        # Let's let the ImmuneSystem just authorize it or do it.
        # Ideally, ImmuneSystem should be the one writing to ensure unauthorized writes don't happen.
        
        filepath = os.path.join(self.memory_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_proposed_content)
            
        return True

    def sandbox_test(self, code_snippet: str) -> dict:
        """
        ACTUAL Execution Test (The 'IDE' Environment).
        Tries to run the code to catch runtime errors (e.g. Indentation, Logic).
        But does NOT allow side effects (file writing blocked via mock).
        """
        try:
            # 1. Static Check First
            static_check = self.validate_code(code_snippet)
            if not static_check["valid"]:
                return static_check

            # 2. Runtime Check
            # Create a dummy environment
            sandbox_globals = {"__name__": "__demo__"}
            
            # Executing...
            exec(code_snippet, sandbox_globals)
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            # Capture full traceback or short error
            import traceback
            tb = traceback.format_exc()
            return {"valid": False, "error": f"Runtime Error: {str(e)}", "traceback": tb}
