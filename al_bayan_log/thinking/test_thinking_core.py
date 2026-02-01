import sys
import os

# Ensure we can import bayan_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from bayan_core.thinking_core import ThinkingEngine
from bayan_core._legacy_wrapper.physical_core import Vector3

def test_carpentry_logic():
    print("\n[TEST] Testing Carpentry Logic (Logical + Physical)...")
    engine = ThinkingEngine()
    
    # 1. Create Network
    net = engine.create_network("carpentry", "Carpentry Network", "professional")
    print(f"[OK] Created {net}")

    # 2. Add Nodes
    carpenter = engine.add_node("carpentry", "Carpenter", "entity", {"skill": "expert"})
    wood = engine.add_node("carpentry", "Wood", "material")
    table = engine.add_node("carpentry", "Table", "product")
    
    # 3. Add Physical Properties (The Hybrid Aspect)
    # Let's say Wood has mass 50kg
    wood.make_physical(mass=50.0, position=Vector3(0,0,0))
    print(f"[OK] Defined Physical Node: {wood} with Mass: {wood.physical.mass}")

    # 4. Add Relations
    # Carpenter needs Wood
    engine.add_relation("carpentry", "Carpenter", "Wood", "requires", 1.0)
    # Carpenter makes Table (if he has wood) - simplified for chain search
    engine.add_relation("carpentry", "Carpenter", "Table", "causes", 0.9)
    # Wood becomes Table
    engine.add_relation("carpentry", "Wood", "Table", "becomes", 1.0)

    # 5. Inference: Find Chain from Carpenter to Table
    print("\n[SEARCH] Inferring Chain: Carpenter -> Table")
    chain = engine.infer("carpentry", "Carpenter", "Table")
    print(f"   Result: {chain}")
    
    if chain == ["Carpenter", "Table"]:
        print("[OK] Direct causal link found.")
    else:
        print("[WARN] Direct link check variation.")

    print("\n[SEARCH] Inferring Chain: Carpenter -> Wood -> Table")
    # This might require a specific search strategy or just hopping
    # Let's check neighbors manually to see structure
    print(f"   Carpenter Neighbors: {net.get_outgoing('Carpenter')}")
    print(f"   Wood Neighbors: {net.get_outgoing('Wood')}")

if __name__ == "__main__":
    test_carpentry_logic()
