from typing import Dict, List, Optional
from .node import BayanNode
from .relation import CausalRelation

class CausalNetwork:
    """
    The container graph that holds nodes and relations.
    """
    def __init__(self, net_id: str, name: str, domain: str = "general"):
        self.id = net_id
        self.name = name
        self.domain = domain
        
        self.nodes: Dict[str, BayanNode] = {}
        # Adjacency list: node_id -> list of outgoing relations
        self.adj: Dict[str, List[CausalRelation]] = {}
        # Reverse adjacency: node_id -> list of incoming relations (for effect->cause search)
        self.rev_adj: Dict[str, List[CausalRelation]] = {}

    def add_node(self, node: BayanNode):
        if node.id not in self.nodes:
            self.nodes[node.id] = node
            self.adj[node.id] = []
            self.rev_adj[node.id] = []

    def add_relation(self, relation: CausalRelation):
        # Ensure nodes exist
        self.add_node(relation.source)
        self.add_node(relation.target)
        
        self.adj[relation.source.id].append(relation)
        self.rev_adj[relation.target.id].append(relation)

    def get_node(self, node_id: str) -> Optional[BayanNode]:
        return self.nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> List[CausalRelation]:
        return self.adj.get(node_id, [])
        
    def get_incoming(self, node_id: str) -> List[CausalRelation]:
        return self.rev_adj.get(node_id, [])

    def __repr__(self):
        return f"<CausalNetwork '{self.name}' ({len(self.nodes)} nodes, {self.domain})>"
