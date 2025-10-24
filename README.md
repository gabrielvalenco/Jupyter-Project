# Delivery Route Optimization with A* (Jupyter Project)

## Overview
This project models a city as a graph where nodes are delivery points and edges are roads with costs (distance, time, or any metric). The goal is to find the lowest-cost tour that starts and ends at a given depot while visiting a set of delivery points. We adapt the A* algorithm to this Traveling Salesman Problem (TSP)-like setting using state `(current_node, visited_set)` and admissible heuristics.

## Features
- Graph-based modeling with weighted edges: Euclidean, Manhattan, or provided road costs.
- Adapted A* over states `(current_node, visited_set)` that returns to the depot.
- Heuristics: MST-based lower bound (recommended), 1-tree/Held–Karp bound (optional), simple straight-line fallback.
- Visualizations with `networkx`/`matplotlib`, optional map with `folium`.
- Pluggable cost metrics; easy to switch between distance/time or custom weights.

## Requirements
- Python 3.10+
- Install the main dependencies:

```
pip install jupyterlab networkx numpy scipy matplotlib pandas tqdm ortools folium
```

## Quick Start
1. Launch Jupyter: `jupyter lab` or `jupyter notebook`.
2. Create or open the notebook (e.g., `DeliveryAStar.ipynb`).
3. Provide inputs:
   - Depot ID (start/end node).
   - List of delivery IDs to visit.
   - Either coordinates (for Euclidean/Manhattan) or an edge list with weights (for road network costs).
4. Run cells to build the graph, select the heuristic, and compute the route.
5. Inspect total cost and visualizations.

## Input Formats
- Coordinates CSV: columns `id,x,y` (or `id,lat,lon`).
- Graph edges CSV: columns `u,v,weight` (undirected or directed; choose accordingly).
- JSON formats can mirror the CSV structure.

## Graph Modeling
- Euclidean: `weight = sqrt((x1-x2)^2 + (y1-y2)^2)`.
- Manhattan: `weight = |x1-x2| + |y1-y2|`.
- Road network: use given `weight` per edge (distance/time/combined cost).

## Algorithm (Adapted A*)
- State: `(current, visited_mask)`, where `visited_mask` tracks depot and deliveries.
- Transitions: move along edges; accumulate `g` (path cost so far).
- Goal: all deliveries visited and `current == depot` (closed tour).
- Priority: `f = g + h`, where `h` is an admissible lower bound on the remaining tour.

### Heuristics
- MST Lower Bound (recommended):
  - `h = cost(MST(remaining ∪ {depot})) + min_d(current, remaining) + min_d(depot, remaining)`.
  - This is admissible and typically consistent; compute the MST with `networkx.minimum_spanning_tree` or `scipy`.
- 1-Tree (Held–Karp) Lower Bound (advanced):
  - Stronger bound using degree penalties and a 1-tree relaxation; consider for harder instances.
- Simple Straight-Line Fallback:
  - e.g., sum of Euclidean/Manhattan distances from remaining points; fast but may not be strictly admissible.
- Sum of remaining distances ("sum"):
  - `h = sum(dist_matrix[current, r] for r in remaining)`; if `remaining` is empty, use `dist_matrix[current, depot]`.
  - Very fast; informative for clustered points. In arbitrary road networks, it may not be strictly admissible, so prefer `mst` when optimality guarantees are needed.
  - Usage (Python): `from src.route_solver import astar_tsp; path, cost = astar_tsp(dist_matrix, depot_index, delivery_indices, heuristic="sum")`.
  - Usage (Notebook): set `heuristic="sum"` in the solver cell.

## Notebook Structure
- `01_problem_and_data.ipynb` — define inputs and parse CSV/JSON.
- `02_graph_model.ipynb` — construct graph and choose cost metric.
- `03_astar_tsp.ipynb` — implement adapted A* with chosen heuristic.
- `04_visualization.ipynb` — draw the route and report statistics.
- `05_benchmarks.ipynb` — compare heuristics and scalability.

Alternatively, use a single notebook `DeliveryAStar.ipynb` containing all sections.

## Performance Notes
- State space is `O(n · 2^n)`; A* reduces exploration via heuristics.
- With MST heuristic, ~10–15 delivery points are practical on a laptop.
- Improve performance with 1-tree bound, pruning, and memoization (bitmask DP caches).

## Alternatives
- Google OR-Tools (TSP/VRP) for larger instances or vehicle fleets.
- Lin–Kernighan–Helsgaun (LKH) for high-quality heuristic tours.
- Metaheuristics (GA, SA, Tabu) when exactness is less critical.

## Visualization
- Use `networkx.draw` to visualize nodes/edges and highlight the tour.
- For geographic data, `folium` can render routes on a map.

## License
See `LICENSE` for licensing information.