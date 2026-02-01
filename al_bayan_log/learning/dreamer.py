# -*- coding: utf-8 -*-
"""
Al-Qalam Dreamer 🛌
Offline optimization module.
Refactors memory, archives unused concepts, and consolidates knowledge.
"""

import os
import sys
import datetime
from typing import List, Dict

# Ensure core modules are available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from librarian import Librarian
from immune_system import ImmuneSystem
from dynamic_cell import DynamicCell

class Dreamer:
    """
    The subconscious of Al-Qalam.
    Runs when the system is idle to clean and optimize memory.
    """
    
    def __init__(self, memory_root):
        self.memory_root = memory_root
        self.librarian = Librarian(memory_root)
        self.immune = ImmuneSystem(memory_root)
        self.archive_file = "archive.py"
        
        # Ensure archive exists
        archive_path = os.path.join(memory_root, self.archive_file)
        if not os.path.exists(archive_path):
            with open(archive_path, 'w', encoding='utf-8') as f:
                f.write(f"# === Al-Qalam Memory Archive ===\n")
                f.write(f"# Items removed from active memory during dreaming\n")
                f.write("from core.dynamic_cell import DynamicCell\n\n")

    def dream(self):
        """Main entry point for the dreaming process."""
        print("🛌 Entering Dream State (Optimization)...")
        
        # 1. Load all active cells
        active_cells = self._load_all_cells()
        print(f"   • Loaded {len(active_cells)} cells from active memory.")
        
        # 2. Identify candidates for archiving (Garbage Collection)
        # Criteria: activation_count == 0
        to_archive = []
        for cell in active_cells.values():
            # Don't archive core concepts or very new ones (less than 1 hour old)
            # For demo functionality, we'll just check activation_count == 0
            if cell.stats.get("activation_count", 0) == 0:
                # Basic check: typically would also check creation time to avoid killing newborns
                to_archive.append(cell)
        
        print(f"   • Found {len(to_archive)} unused cells candidates for archiving.")
        
        # 3. Move them to archive
        if to_archive:
            self._archive_cells(to_archive)
            
        print("✨ Dream Cycle Complete.")

    def _load_all_cells(self) -> Dict[str, DynamicCell]:
        """Loads all cells from active memory books (excluding archive)."""
        all_cells = {}
        for filename in self.librarian.books.values():
            if filename == self.archive_file: 
                continue
                
            path = os.path.join(self.memory_root, filename)
            if not os.path.exists(path):
                continue
                
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Prepare execution environment
                local_vars = {}
                global_vars = globals().copy() 
                global_vars['DynamicCell'] = DynamicCell
                
                # Verify that 'core' is importable or mock it
                # The generated files might do: from core.dynamic_cell import DynamicCell
                # We need to make sure that works.
                # Simplest hack: Map 'core.dynamic_cell' to current module content if using exec
                
                try:
                    exec(code, global_vars, local_vars)
                except ImportError:
                    # Retry with modified sys.path or patching source
                    # If 'from core.dynamic_cell' fails, maybe we are running from root?
                    # Let's just strip 'core.' from the source code in memory before exec
                    # This is safer than messing with sys.modules
                    code_patched = code.replace("from core.dynamic_cell", "from dynamic_cell")
                    exec(code_patched, global_vars, local_vars)
                
                for k, v in local_vars.items():
                    if isinstance(v, DynamicCell):
                        # Tag source file for removal later
                        v._source_file = filename 
                        all_cells[v.id] = v
            except Exception as e:
                print(f"   ⚠️ Error reading {filename}: {e}")
                
        return all_cells

    def _archive_cells(self, cells: List[DynamicCell]):
        """Moves cells from their source files to the archive."""
        
        # Group by source file to minimize file IO
        by_file = {}
        for cell in cells:
            src = getattr(cell, '_source_file', None)
            if src:
                by_file.setdefault(src, []).append(cell)
                
        # Process each file
        for filename, cell_list in by_file.items():
            print(f"   📦 Archiving {len(cell_list)} cells from {filename}...")
            
            # Read source
            path = os.path.join(self.memory_root, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove cells from source content
            # This is tricky with text replacement. 
            # Strategy: Re-generate the file WITHOUT these cells.
            # Robust way: Load ALL cells from that file, filter out archived ones, rewrite file.
            
            self._rewrite_file_without(filename, [c.id for c in cell_list])
            
            # Append to archive
            archive_code = ""
            for cell in cell_list:
                archive_code += f"\n# Archived at {datetime.datetime.now().isoformat()}\n"
                archive_code += cell.to_source_code() + "\n"
                
            archive_path = self.archive_file
            
            # Use safe write for archive (append)
            full_archive_path = os.path.join(self.memory_root, archive_path)
            existing_archive = ""
            if os.path.exists(full_archive_path):
                 with open(full_archive_path, 'r', encoding='utf-8') as f:
                     existing_archive = f.read()
            
            self.immune.safe_write(archive_path, existing_archive, archive_code)
            
    def _rewrite_file_without(self, filename: str, cell_ids_to_remove: List[str]):
        """Rewrites a memory file excluding specific cell IDs."""
        path = os.path.join(self.memory_root, filename)
        
        # Load everything in that file
        with open(path, 'r', encoding='utf-8') as f:
             code = f.read()
        
        local_vars = {}
        global_vars = globals().copy()
        global_vars['DynamicCell'] = DynamicCell
        
        try:
             exec(code, global_vars, local_vars)
        except ImportError:
             code_patched = code.replace("from core.dynamic_cell", "from dynamic_cell")
             exec(code_patched, global_vars, local_vars)
        
        kept_cells = []
        for k, v in local_vars.items():
            if isinstance(v, DynamicCell):
                if v.id not in cell_ids_to_remove:
                    kept_cells.append(v)
                    
        # Regenerate file content
        new_content = f"# Memory File: {filename}\n"
        new_content += "try:\n    from core.dynamic_cell import DynamicCell\nexcept ImportError:\n    from dynamic_cell import DynamicCell\n\n"
        
        for cell in kept_cells:
            new_code = cell.to_source_code()
            new_content += new_code + "\n"
            
        # Write back (Overwrite)
        # Note: safe_write typically appends or checks merge. Here we are replacing.
        # Ideally ImmuneSystem should handle "Rewrite" operations too.
        # For now, we'll bypass safe_write strictly for rewrite to avoid "duplicate definition" errors if we append
        # But we should back it up!
        
        self.immune.backup_memory(filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
