# Delivery Route Optimization with A* (Jupyter Project)

## Overview
Model a city as a graph with delivery points (nodes) and roads (edges with costs). Find a lowest-cost tour that starts and ends at a depot while visiting all specified deliveries. We adapt A* to a TSP-like search over `(current_node, visited_set)` states.

## Features
- Graph-based modeling with weighted edges: Euclidean, Manhattan, or provided road costs.
- Adapted A* over `(current_node, visited_set)` that returns to the depot.
- Heuristics: MST-based lower bound (recommended), 1-tree/Held–Karp (optional), simple straight-line fallback.
- Visualizations with `networkx` and `matplotlib`; optional `folium` for maps.

## Requirements
- Python 3.10+
- Install: `pip install -r requirements.txt`

## Quick Start
- Launch Jupyter: `jupyter lab` or `jupyter notebook`.
- Open `notebooks/DeliveryAStar.ipynb` and run the cells.
- Inputs: depot ID, delivery list, coordinates or edge list with weights.

## Algorithm
- State: `(current, visited_mask)`; `f = g + h` with admissible `h`.
- Goal: all deliveries visited and `current == depot` (return to depot).
- MST heuristic: `MST(remaining ∪ {depot}) + min(current→remaining) + min(depot→remaining)`.

## Structure
- `src/route_solver.py`: distance matrix, MST heuristic, A* search.
- `data/sample_points.csv`: sample depot and deliveries.
- `notebooks/DeliveryAStar.ipynb`: demo with visualization.

## License
See `LICENSE` for details.