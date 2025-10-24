import nbformat as nbf
import uuid

NOTEBOOK_PATH = r"notebooks/DeliveryAStar.ipynb"


VALIDATION_CODE = """
import itertools

def tour_cost(order):
    c = 0.0
    prev = depot_idx
    for j in order:
        c += float(dist[prev, j])
        prev = j
    c += float(dist[prev, depot_idx])
    return c

# Compute best cost and capture all optimal orders (to handle ties)
best_cost = float('inf')
optimal_orders = []
for order in itertools.permutations(delivery_indices):
    c = tour_cost(order)
    if c < best_cost - 1e-12:
        best_cost = c
        optimal_orders = [order]
    elif abs(c - best_cost) <= 1e-9:
        optimal_orders.append(order)

def labels_from_order(order):
    return [labels[depot_idx]] + [labels[i] for i in order] + [labels[depot_idx]]

def cycles_equivalent(t1, t2):
    # both include depot at start and end
    a = t1[1:-1]
    b = t2[1:-1]
    if len(a) != len(b):
        return False
    n = len(a)
    for k in range(n):
        rot = a[k:] + a[:k]
        if rot == b or rot[::-1] == b:
            return True
    return False

astar_labels = [labels[i] for i in path_indices]
found_equiv = any(cycles_equivalent(astar_labels, labels_from_order(o)) for o in optimal_orders)

print("Optimal cost match:", abs(total_cost - best_cost) <= 1e-9)
print("Route equivalent to an optimal:", found_equiv)
print("Best cost:", best_cost)
print("One optimal tour:", labels_from_order(optimal_orders[0]))
""".strip()


def ensure_win_loop_cell(nb):
    # Insert WindowsSelectorEventLoopPolicy early to avoid RuntimeWarning on Windows
    code = (
        "import sys, asyncio\n"
        "if sys.platform.startswith('win'):\n"
        "    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())\n"
    )
    for c in nb.cells:
        if c.cell_type == "code" and "WindowsSelectorEventLoopPolicy" in (c.source or ""):
            return False
    # Insert right after the title markdown (index 1) if exists
    insert_idx = 1 if nb.cells and nb.cells[0].cell_type == "markdown" else 0
    nb.cells.insert(insert_idx, nbf.v4.new_code_cell(code))
    return True


def ensure_validation_cell(nb):
    # Add validation cell after A* run cell if not present
    inserted = False
    already = False
    for c in nb.cells:
        if c.cell_type == "code" and c.source and "cycles_equivalent" in c.source:
            already = True
            break
    if already:
        return False
    for idx, c in enumerate(nb.cells):
        if c.cell_type == "code" and "path_indices, total_cost = astar_tsp" in c.source:
            nb.cells.insert(idx + 1, nbf.v4.new_code_cell(VALIDATION_CODE))
            inserted = True
            break
    return inserted


def enhance_visualization(nb):
    changed = False
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        src = c.source or ""
        if "plt.title('A* MST Tour')" in src:
            src = src.replace(
                "plt.title('A* MST Tour')",
                "plt.title(f'A* MST Tour (cost={total_cost:.3f})')",
            )
            changed = True
        if "nx.draw(G, pos" in src and "draw_networkx_nodes" not in src:
            src = src.replace(
                "nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=600)",
                "nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=600)\n"
                "nx.draw_networkx_nodes(G, pos, nodelist=[depot_idx], node_color='gold', node_size=800)",
            )
            changed = True
        c.source = src
    return changed


def add_missing_ids(nb):
    changed = False
    for c in nb.cells:
        if not getattr(c, 'id', None):
            c['id'] = uuid.uuid4().hex
            changed = True
    return changed


def fix_root_cell(nb):
    desired = (
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd().resolve()\n"
        "if not (ROOT / 'src').exists():\n"
        "    ROOT = ROOT.parent\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.append(str(ROOT))\n"
        "print('Python:', sys.executable)\n"
    )
    for c in nb.cells:
        if c.cell_type == "code" and c.source and "from pathlib import Path" in c.source and "ROOT =" in c.source:
            c.source = desired
            return True
    return False


def main():
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)
    win_loop_added = ensure_win_loop_cell(nb)
    ids_added = add_missing_ids(nb)
    root_fixed = fix_root_cell(nb)
    v_added = ensure_validation_cell(nb)
    vis_changed = enhance_visualization(nb)
    if win_loop_added or ids_added or root_fixed or v_added or vis_changed:
        nbf.write(nb, NOTEBOOK_PATH)
        print("Notebook updated:", {
            "win_loop_added": win_loop_added,
            "ids_added": ids_added,
            "root_fixed": root_fixed,
            "validation_cell_added": v_added,
            "visualization_enhanced": vis_changed,
        })
    else:
        print("No changes applied (already up-to-date).")


if __name__ == "__main__":
    main()