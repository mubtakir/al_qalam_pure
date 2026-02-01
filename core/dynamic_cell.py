#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynamicCell: A neuron that can serialize its state into executable Python code.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import hashlib

class AdaptiveParameter:
    """A dynamic parameter that calculates its value from a base and a delta."""
    def __init__(self, base: float, delta: float = 0.0, history: Optional[List[float]] = None):
        self.base = base
        self.delta = delta
        self.history = history or []

    @property
    def value(self) -> float:
        return self.base + self.delta

    def to_dict(self) -> Dict:
        return {"base": self.base, "delta": self.delta, "history": self.history}

class DynamicCell:
    """A neural cell with self-serialization and state-tracking capabilities."""
    
    def __init__(self, 
                 cell_id: str,
                 cell_type: str = "neuron",
                 initial_value: float = 0.0,
                 metadata: Optional[Dict] = None):
        
        self.id = cell_id
        self.type = cell_type
        self.value = initial_value
        self.activation = 0.0
        self.connections: Dict[str, AdaptiveParameter] = {}  # {target_id: AdaptiveParameter}
        
        self.metadata = metadata or {}
        self.metadata.setdefault("created", datetime.now().isoformat())
        self.metadata["last_modified"] = datetime.now().isoformat()
        
        self.stats = {
            "activation_count": 0,
            "total_activation": 0.0
        }
        self.memory: List[float] = []

    def activate(self, input_value: float) -> float:
        """Standard sigmoid activation function."""
        combined = input_value + self.value
        self.activation = 1 / (1 + np.exp(-combined))
        
        # Track stats
        self.stats["activation_count"] += 1
        self.stats["total_activation"] += self.activation
        if len(self.memory) < 100:
            self.memory.append(self.activation)
            
        return self.activation

    def connect_to(self, target_id: str, weight: float = 0.5, delta: float = 0.0):
        """Establish or update an adaptive connection."""
        if target_id in self.connections:
            self.connections[target_id].base = weight
            self.connections[target_id].delta = delta
        else:
            self.connections[target_id] = AdaptiveParameter(weight, delta)
        self.metadata["last_modified"] = datetime.now().isoformat()

    def to_source_code(self) -> str:
        """Serializes the current cell state into an executable Python snippet."""
        meta_json = json.dumps(self.metadata, indent=2, ensure_ascii=False)
        stats_json = json.dumps(self.stats, indent=2)
        memory_json = json.dumps(self.memory)
        
        code = f"""
# --- DYNAMIC CELL: {self.id} ---
cell_{self.id} = DynamicCell(
    cell_id="{self.id}",
    cell_type="{self.type}",
    initial_value={self.value},
    metadata={meta_json}
)
cell_{self.id}.stats = {stats_json}
cell_{self.id}.memory = {memory_json}
"""
        # Add connections as adaptive blocks
        for target, param in self.connections.items():
            code += f"cell_{self.id}.connect_to('{target}', weight={param.base}, delta={param.delta}) # adaptive_slot\n"
            
        return code
