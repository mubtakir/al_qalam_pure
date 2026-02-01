from typing import List, Dict, Set, Optional
from .network import CausalNetwork
from .relation import CausalRelation

def find_causal_chain(net: CausalNetwork, start_id: str, end_id: str, max_depth: int = 5) -> List[str]:
    """
    Finds a path from start_node to end_node using BFS.
    Returns: List of node IDs representing the path, or empty list if none found.
    """
    if start_id not in net.nodes or end_id not in net.nodes:
        return []
        
    queue = [(start_id, [start_id])] # (current_id, path_so_far)
    visited = {start_id}
    
    while queue:
        curr_id, path = queue.pop(0)
        
        if len(path) > max_depth + 1:
            continue
            
        if curr_id == end_id:
            return path
            
        for rel in net.get_outgoing(curr_id):
            neighbor = rel.target.id
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    return []

def find_all_effects(net: CausalNetwork, start_id: str, max_depth: int = 2) -> Dict[str, List[str]]:
    """
    Returns all reachable nodes (effects) up to max_depth.
    Returns: Dict[depth, list_of_node_ids]
    """
    results = {}
    if start_id not in net.nodes:
        return results
        
    queue = [(start_id, 0)]
    visited = {start_id}
    
    while queue:
        curr_id, depth = queue.pop(0)
        
        if depth > 0:
            if depth not in results: results[depth] = []
            results[depth].append(curr_id)
            
        if depth >= max_depth:
            continue
            
        for rel in net.get_outgoing(curr_id):
            neighbor = rel.target.id
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
                
    return results

def predict_outcome(net: CausalNetwork, initial_state: str, action: str) -> str:
    """
    Simple forward prediction: 
    If initial_state -> (rel: causes/enables) -> X.
    """
    # This is a simplified placeholder. Real prediction requires state simulation.
    # Look for relations from initial_state that match the action context
    outgoing = net.get_outgoing(initial_state)
    for rel in outgoing:
        # If the relation type matches implied causality or specific action logic
        if rel.type in ["causes", "leads_to", "results_in"]:
             return f"Outcome: {rel.target.id} (via {rel.type})"
             
    return "No obvious immediate outcome found."
