from typing import Dict, Any, List
from .network import CausalNetwork
from .node import BayanNode
from .relation import CausalRelation
from .inference import find_causal_chain, find_all_effects, predict_outcome

class ThinkingEngine:
    """
    The High-Level Procedural Interface for the Logical Brain.
    Acts as the API for creating/managing networks and running inference.
    """
    def __init__(self):
        self.networks: Dict[str, CausalNetwork] = {}
        self.active_net_id: str = None
        
    def create_network(self, net_id: str, name: str, domain: str) -> CausalNetwork:
        net = CausalNetwork(net_id, name, domain)
        self.networks[net_id] = net
        self.active_net_id = net_id
        return net
        
    def get_network(self, net_id: str) -> CausalNetwork:
        return self.networks.get(net_id)

    def add_node(self, net_id: str, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        net = self.get_network(net_id)
        if net:
            node = BayanNode(node_id, node_type, properties)
            net.add_node(node)
            return node
        raise ValueError(f"Network {net_id} not found")

    def add_relation(self, net_id: str, src: str, dst: str, rel: str, strength: float = 1.0):
        net = self.get_network(net_id)
        if net:
            s_node = net.get_node(src)
            d_node = net.get_node(dst)
            if s_node and d_node:
                relation = CausalRelation(s_node, d_node, rel, strength)
                net.add_relation(relation)
                return relation
            raise ValueError(f"Nodes {src} or {dst} not found in {net_id}")
        raise ValueError(f"Network {net_id} not found")

    def infer(self, net_id: str, start: str, end: str):
        net = self.get_network(net_id)
        if net:
            return find_causal_chain(net, start, end)
        return []

    # --- CLI / Debug Output ---
    def print_network_summary(self, net_id: str):
        net = self.get_network(net_id)
        if not net:
            print(f"Network {net_id} not found.")
            return
            
        print(f"=== Network: {net.name} ({net.domain}) ===")
        print(f"Nodes: {len(net.nodes)}")
        print(f"Relations: {sum(len(l) for l in net.adj.values())}")
