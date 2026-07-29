from collections import defaultdict
from routing.bundle_generation import BundleGenerator
from routing.utility import UtilityModel

def _undirected_edge(edge):
    return tuple(sorted(edge))


def shortest_feasible_path(bundles, request_id):
    eligible = [b for b in bundles if b["request_id"] == request_id]
    if not eligible:
        return None
    path_lens = {}
    for b in eligible:
        pl = sum(
            1 for _ in b["path"]
        ) - 1
        path_lens[b["bundle_id"]] = pl
    min_len = min(path_lens.values())
    shortest = [b for b in eligible if sum(1 for _ in b["path"]) - 1 == min_len]
    return max(shortest, key=lambda b: b["utility"])


def highest_fidelity_path(bundles, request_id):
    eligible = [b for b in bundles if b["request_id"] == request_id]
    if not eligible:
        return None
    from routing.fidelity import FidelityModel
    best = None
    best_fid = -1
    for b in eligible:
        if b.get("fidelity", 0) > best_fid:
            best_fid = b.get("fidelity", 0)
            best = b
    return best


def utility_density_greedy(bundles, edge_capacities, memory_capacities):
    ec = {_undirected_edge(k): v for k, v in edge_capacities.items()}
    mc = memory_capacities

    scored = []
    for b in bundles:
        total_demand = sum(b["edge_demands"].values()) + sum(b["memory_demands"].values())
        density = b["utility"] / (total_demand + 1e-10)
        scored.append((density, b))
    scored.sort(reverse=True)

    selections = {}
    assigned_requests = set()
    for _, b in scored:
        rid = b["request_id"]
        if rid in assigned_requests:
            continue
        feasible = True
        for edge, d in b["edge_demands"].items():
            if d > ec.get(_undirected_edge(edge), 0):
                feasible = False
                break
        if feasible:
            for node, d in b["memory_demands"].items():
                if d > mc.get(node, 0):
                    feasible = False
                    break
        if feasible:
            selections[rid] = b["bundle_id"]
            assigned_requests.add(rid)

    for b in bundles:
        rid = b["request_id"]
        if rid not in selections:
            selections[rid] = None

    selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
    return {"selected": selected, "selections": selections}


def fidelity_aware_greedy(bundles, edge_capacities, memory_capacities):
    ec = {_undirected_edge(k): v for k, v in edge_capacities.items()}
    mc = memory_capacities

    scored = []
    for b in bundles:
        score = b["utility"]
        scored.append((score, b))
    scored.sort(reverse=True)

    selections = {}
    assigned_requests = set()
    for _, b in scored:
        rid = b["request_id"]
        if rid in assigned_requests:
            continue
        feasible = True
        for edge, d in b["edge_demands"].items():
            if d > ec.get(_undirected_edge(edge), 0):
                feasible = False
                break
        if feasible:
            for node, d in b["memory_demands"].items():
                if d > mc.get(node, 0):
                    feasible = False
                    break
        if feasible:
            selections[rid] = b["bundle_id"]
            assigned_requests.add(rid)

    for b in bundles:
        rid = b["request_id"]
        if rid not in selections:
            selections[rid] = None

    selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
    return {"selected": selected, "selections": selections}
