#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InferenceRule: A class representing a logical if-then rule that persists as source code.
"""

from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

class InferenceRule:
    """A logical rule that can be triggered by context and serialize itself."""
    
    def __init__(self, 
                 rule_id: str,
                 description: str,
                 condition_func_code: str,
                 action_func_code: str,
                 metadata: Optional[Dict] = None):
        
        self.id = rule_id
        self.description = description
        self.condition_code = condition_func_code
        self.action_code = action_func_code
        self.metadata = metadata or {}
        self.metadata.setdefault("created", datetime.now().isoformat())
        self.usage_count = 0

    def to_source_code(self) -> str:
        """Serializes the rule into a robust Python function block."""
        code = f"""
# --- INFERENCE RULE: {self.id} ---
# Description: {self.description}
def rule_{self.id}(model, context):
    \"\"\"Generated logical rule: {self.description}\"\"\"
    
    # Condition Logic
    def check_condition(model, context):
{self._indent(self.condition_code)}
        
    # Action Logic
    def perform_action(model, context):
{self._indent(self.action_code)}
    
    if check_condition(model, context):
        perform_action(model, context)
        return True
    return False

# Registering metadata for {self.id}
rule_{self.id}._metadata = {self.metadata}
"""
        return code

    def _indent(self, code: str, spaces: int = 8) -> str:
        lines = code.strip().split('\n')
        return '\n'.join([' ' * spaces + line for line in lines])
