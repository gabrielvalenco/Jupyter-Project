import math
import heapq
from typing import List, Tuple, Callable, Dict

import numpy as np
import networkx as nx


def build_distance_matrix(points: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    n = len(points)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0.0
            else:
                if metric == "euclidean":
                    dx = points[i][0] - points[j][0]
                    dy = points[i][1] - points[j][1]
                    dist[i, j] = math.hypot(dx, dy)
                elif metric == "manhattan":
                    dx = abs(points[i][0] - points[j][0])
                    dy = abs(points[i][1] - points[j][1])
                    dist[i, j] = dx + dy
                else:
                    raise ValueError(f"Unknown metric: {metric}")
    return dist


def _mst_cost(nodes: List[int], dist_matrix: np.ndarray) -> float:
    """Compute MST cost over a complete graph induced by nodes using dist_matrix as weights."""
    if not nodes:
        return 0.0
    G = nx.Graph()
    for i in nodes:
        G.add_node(i)
    for idx, u in enumerate(nodes):
        for v in nodes[idx + 1 :]:
            G.add_edge(u, v, weight=float(dist_matrix[u, v]))
    T = nx.minimum_spanning_tree(G, weight="weight")
    return float(sum(d["weight"] for _, _, d in T.edges(data=True)))


def heuristic_mst(current: int, remaining: List[int], dist_matrix: np.ndarray, depot: int) -> float:
    """
    Admissible MST-based lower bound:
    h = MST(remaining ∪ {depot}) + min_d(current, remaining) + min_d(depot, remaining)
    If no remaining, return distance to depot.
    """
    if not remaining:
        return float(dist_matrix[current, depot])
    base_nodes = list(remaining)
    if depot not in base_nodes:
        base_nodes.append(depot)
    mst = _mst_cost(base_nodes, dist_matrix)
    to_current = min(dist_matrix[current, r] for r in remaining)
    to_depot = min(dist_matrix[depot, r] for r in remaining)
    return float(mst + to_current + to_depot)


def astar_tsp(
    dist_matrix: np.ndarray,
    depot_index: int,
    delivery_indices: List[int],
    heuristic: str = "mst",
) -> Tuple[List[int], float]:
    """
    A* over state (current_node, visited_mask_of_deliveries) on a complete graph implicit in dist_matrix.
    Returns (path_indices, total_cost). Path starts and ends at depot.
    """
    n_deliv = len(delivery_indices)
    full_mask = (1 << n_deliv) - 1

    # Map delivery node -> bit index
    delivery_to_bit: Dict[int, int] = {node: i for i, node in enumerate(delivery_indices)}

    start_state = (depot_index, 0)

    def h_func(current: int, mask: int) -> float:
        remaining = [d for d in delivery_indices if not (mask & (1 << delivery_to_bit[d]))]
        if heuristic == "mst":
            return heuristic_mst(current, remaining, dist_matrix, depot_index)
        # Simple fallback: sum of nearest distances (fast, not always admissible)
        if not remaining:
            return float(dist_matrix[current, depot_index])
        return float(min(dist_matrix[current, r] for r in remaining) + sum(
            min(dist_matrix[r, depot_index], min(dist_matrix[r, r2] for r2 in remaining if r2 != r))
            for r in remaining
        ))

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    g_score: Dict[Tuple[int, int], float] = {start_state: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    heapq.heappush(open_heap, (h_func(*start_state), 0.0, start_state))

    visited_best_f: Dict[Tuple[int, int], float] = {}

    while open_heap:
        f, g, (current, mask) = heapq.heappop(open_heap)
        # Goal: all deliveries visited and we're back at depot
        if mask == full_mask and current == depot_index:
            # reconstruct path
            path_states = [(current, mask)]
            st = (current, mask)
            while st in came_from:
                st = came_from[st]
                path_states.append(st)
            path_states.reverse()
            path_indices = [s[0] for s in path_states]
            return path_indices, g_score[(current, mask)]

        # Prune if we've already seen a better f for this state
        prev_best = visited_best_f.get((current, mask))
        if prev_best is not None and prev_best < f - 1e-12:
            continue
        visited_best_f[(current, mask)] = f

        # Expand neighbors
        if mask != full_mask:
            # go to any unvisited delivery
            for nxt in delivery_indices:
                bit = 1 << delivery_to_bit[nxt]
                if not (mask & bit):
                    new_mask = mask | bit
                    tentative_g = g_score[(current, mask)] + float(dist_matrix[current, nxt])
                    state = (nxt, new_mask)
                    if tentative_g < g_score.get(state, float("inf")):
                        g_score[state] = tentative_g
                        came_from[state] = (current, mask)
                        f_score = tentative_g + h_func(nxt, new_mask)
                        heapq.heappush(open_heap, (f_score, tentative_g, state))
        else:
            # all deliveries visited; return to depot
            nxt = depot_index
            tentative_g = g_score[(current, mask)] + float(dist_matrix[current, nxt])
            state = (nxt, mask)
            if tentative_g < g_score.get(state, float("inf")):
                g_score[state] = tentative_g
                came_from[state] = (current, mask)
                f_score = tentative_g + h_func(nxt, mask)
                heapq.heappush(open_heap, (f_score, tentative_g, state))

    raise RuntimeError("A* search failed to find a solution. Check inputs.")