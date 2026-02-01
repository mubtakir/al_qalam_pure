from typing import Dict, Any, Optional
import time
from .._legacy_wrapper.physical_core import PhysicalEntity, PhysicalQuantity, Vector3, Dimension

class BayanNode:
    """
    Represents a fundamental unit in the Causal Network.
    Integrates Logical (type, properties) and Physical (mass, velocity, etc.) aspects.
    """
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        self.id = node_id
        self.type = node_type
        self.properties = properties or {}
        self.creation_time = time.time()
        
        # Physical Aspect (Optional)
        self.physical: Optional[PhysicalEntity] = None

    def make_physical(self, mass: float = 1.0, position: Vector3 = None, velocity: Vector3 = None):
        """
        Elevates this node to a physical entity.
        """
        if position is None: position = Vector3(0,0,0)
        if velocity is None: velocity = Vector3(0,0,0)
        
        # Create Physical Quantity wrappers
        mass_pq = PhysicalQuantity(mass, Dimension(M=1), "kg")
        pos_pq = PhysicalQuantity(position, Dimension(L=1), "m")
        vel_pq = PhysicalQuantity(velocity, Dimension(L=1, T=-1), "m/s")
        
        self.physical = PhysicalEntity(self.id, mass_pq, pos_pq, vel_pq)
        self.type = "physical_entity" # Update type if needed

    def __repr__(self):
        phy_tag = " [Physical]" if self.physical else ""
        return f"<Node({self.id}): {self.type}{phy_tag}>"
