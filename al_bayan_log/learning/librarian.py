# -*- coding: utf-8 -*-
"""
Al-Qalam Librarian 📚
Manages the distributed memory system (The Library).
Decides where to file new concepts and retrieving them.
"""

import os
import shutil
from typing import Dict, List, Optional

class Librarian:
    """
    The intelligent file manager for Al-Qalam.
    Routes concepts to specific 'books' (files) based on their nature.
    """
    
    def __init__(self, memory_root_dir):
        self.memory_root = memory_root_dir
        self.books = {
            "entities": "entities.py",   # Physical things (Robot, Apple)
            "actions": "actions.py",     # Verbs (Eats, Loves)
            "emotions": "emotions.py",   # Feelings (Sad, Happy)
            "general": "general.py"      # Everything else
        }
        self._init_library()
        
    def _init_library(self):
        """Ensures all memory 'books' exist."""
        os.makedirs(self.memory_root, exist_ok=True)
        
        # Create __init__.py to make it a package
        init_file = os.path.join(self.memory_root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write("# Al-Qalam Memory Library\n")

        # Initialize books
        for book_name, filename in self.books.items():
            path = os.path.join(self.memory_root, filename)
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(f"# === Al-Qalam Memory: {book_name.upper()} ===\n")
                    f.write("from core.dynamic_cell import DynamicCell\n\n")

    def classify_concept(self, concept_name: str, examples: List[str]) -> str:
        """
        Decides which 'book' a concept belongs to.
        Simple heuristic for now, can be an AI model later.
        """
        # Heuristics
        actions = ["eats", "loves", "runs", "reads", "writes", "يأكل", "يحب", "يشعر"]
        emotions = ["sad", "happy", "angry", "حزين", "سعيد", "غاضب", "الحزن"]
        
        lower_name = concept_name.lower()
        
        if any(act in lower_name for act in actions):
            return "actions"
        if any(emo in lower_name for emo in emotions):
            return "emotions"
            
        # Default assumption: It's an entity if not an action/emotion
        return "entities"

    def get_file_path(self, category: str) -> str:
        """Returns the absolute path for a category's file."""
        filename = self.books.get(category, "general.py")
        return os.path.join(self.memory_root, filename)

    def locate_concept_file(self, concept_name: str) -> Optional[str]:
        """
        Searches where a concept is currently stored.
        Returns filename relative to memory_root or None.
        """
        # This is a naive search. In Phase 2 optimization we'd use an index.
        for filename in self.books.values():
            path = os.path.join(self.memory_root, filename)
            if not os.path.exists(path):
                continue
                
            with open(path, 'r', encoding='utf-8') as f:
                if f"concept_{concept_name}_" in f.read():
                    return filename
        return None
