import random
import networkx as nx
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple


def generate_chain_topology(n_nodes: int, edge_capacity: int = 6,
                            memory_capacity: int = 10,
                            raw_fidelity: float = 0.85,
                            generation_prob: float = 1.0,
                            latency: float = 5.0):
    labels = [f"N{i}" for i in range(n_nodes)]
    edges = [(labels[i], labels[i + 1]) for i in range(n_nodes - 1)]
    edge_caps = {e: edge_capacity for e in edges}
    mem_caps = {n: memory_capacity for n in labels}
    link_params = {}
    for u, v in edges:
        link_params[(u, v)] = {
            "raw_fidelity": raw_fidelity,
            "generation_probability": generation_prob,
            "latency": latency,
        }
    return {
        "nodes": labels,
        "edges": edges,
        "edge_capacities": edge_caps,
        "memory_capacities": mem_caps,
        "link_params": link_params,
    }


def generate_grid_topology(rows: int, cols: int, edge_capacity: int = 6,
                           memory_capacity: int = 10,
                           raw_fidelity: float = 0.85,
                           generation_prob: float = 1.0,
                           latency: float = 5.0):
    labels = [[f"G{r}_{c}" for c in range(cols)] for r in range(rows)]
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((labels[r][c], labels[r][c + 1]))
            if r + 1 < rows:
                edges.append((labels[r][c], labels[r + 1][c]))
    flat_labels = [l for row in labels for l in row]
    edge_caps = {e: edge_capacity for e in edges}
    mem_caps = {n: memory_capacity for n in flat_labels}
    link_params = {}
    for u, v in edges:
        link_params[(u, v)] = {
            "raw_fidelity": raw_fidelity,
            "generation_probability": generation_prob,
            "latency": latency,
        }
    return {
        "nodes": flat_labels,
        "edges": edges,
        "edge_capacities": edge_caps,
        "memory_capacities": mem_caps,
        "link_params": link_params,
    }


def _shortest_paths(topology: dict, source: str, target: str, k: int = 3):
    G = nx.Graph()
    G.add_nodes_from(topology["nodes"])
    G.add_edges_from(topology["edges"])
    try:
        return list(nx.shortest_simple_paths(G, source, target))[:k]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _pareto_prune(bundles, fidelity_key="fidelity",
                  cost_key="bell_pair_cost",
                  latency_key="latency"):
    pruned = []
    for i, bc in enumerate(bundles):
        dominated = False
        for j, bo in enumerate(bundles):
            if i == j:
                continue
            no_worse = (
                bo[fidelity_key] >= bc[fidelity_key]
                and bo[cost_key] <= bc[cost_key]
                and bo[latency_key] <= bc[latency_key]
            )
            strict = (
                bo[fidelity_key] > bc[fidelity_key]
                or bo[cost_key] < bc[cost_key]
                or bo[latency_key] < bc[latency_key]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            pruned.append(bc)
    return pruned


def generate_request_bundles(topology: dict, source: str, target: str,
                             weight: float, min_fidelity: float,
                             q_values: Optional[List[int]] = None) -> List[dict]:
    if q_values is None:
        q_values = [0, 1, 2]
    paths = _shortest_paths(topology, source, target, k=3)
    bundles = []
    for path in paths:
        for q in q_values:
            bundle = _evaluate_bundle(topology, path, q, source, target,
                                      weight, min_fidelity)
            if bundle:
                bundles.append(bundle)
    return _pareto_prune(bundles)


def _evaluate_bundle(topology, path, q, source, target,
                     weight, min_fidelity):
    if len(path) < 2:
        return None
    link_fids = []
    link_succ = []
    total_lat = 0.0
    total_cost = 0
    edge_demands = {}

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = (u, v) if u < v else (v, u)
        lp = topology["link_params"].get(edge, topology["link_params"].get((v, u)))
        if lp is None:
            return None
        f = lp["raw_fidelity"]
        for _ in range(q):
            if f <= 0.5:
                break
            num = f ** 2 + ((1 - f) / 3) ** 2
            den = f ** 2 + 2 * f * (1 - f) / 3 + 5 * ((1 - f) / 3) ** 2
            f = num / den
        link_fids.append(f)
        g = lp["generation_probability"]
        p_succ = g ** (2 ** q)
        ff = lp["raw_fidelity"]
        for rnd in range(q):
            if ff <= 0.5:
                break
            s = ff ** 2 + 2 * ff * (1 - ff) / 3 + 5 * ((1 - ff) / 3) ** 2
            p_succ *= s ** (2 ** (q - rnd - 1))
            num = ff ** 2 + ((1 - ff) / 3) ** 2
            ff = num / s
        link_succ.append(p_succ)
        cost = 2 ** q
        edge_demands[edge] = cost
        total_cost += cost
        total_lat += lp["latency"] * cost

    if not link_fids:
        return None
    end_f = link_fids[0]
    for nf in link_fids[1:]:
        end_f = end_f * nf + (1 - end_f) * (1 - nf) / 3

    if end_f < min_fidelity:
        return None

    succ = 1.0
    for p in link_succ:
        succ *= p

    mem_demands = {}
    for edge, d in edge_demands.items():
        u, v = edge
        mem_demands[u] = mem_demands.get(u, 0) + d
        mem_demands[v] = mem_demands.get(v, 0) + d

    fidelity_margin = max(0.0, end_f - min_fidelity)
    utility = weight * succ * (1.0 + fidelity_margin) - 0.01 * total_lat - 0.05 * total_cost

    return {
        "bundle_id": f"b_{source}_{target}_q{q}",
        "request_id": f"req_{source}_{target}",
        "path": path,
        "edge_demands": edge_demands,
        "memory_demands": mem_demands,
        "utility": utility,
        "latency": total_lat,
        "fidelity": end_f,
        "success_probability": succ,
        "bell_pair_cost": total_cost,
    }


def generate_benchmark_instance(topology: dict,
                                request_pairs: List[Tuple[str, str, float, float]],
                                rng: random.Random) -> Tuple[List[dict], dict, dict]:
    all_bundles = []
    for i, (src, dst, weight, min_fid) in enumerate(request_pairs):
        bundles = generate_request_bundles(topology, src, dst, weight, min_fid)
        for j, b in enumerate(bundles):
            b["request_id"] = f"req_{i}"
            b["bundle_id"] = f"req_{i}_b{j}"
        all_bundles.extend(bundles)
    return all_bundles, topology["edge_capacities"], topology["memory_capacities"]


def contention_sweep_instances(topology_fn: Callable, n_requests_list: List[int],
                               seed: int = 42) -> Dict[str, dict]:
    base = topology_fn()
    rng = random.Random(seed)
    nodes = base["nodes"]
    instances = {}
    for n_req in n_requests_list:
        pairs = []
        for _ in range(n_req):
            src, dst = rng.sample(nodes, 2)
            w = rng.uniform(10.0, 100.0)
            mf = rng.uniform(0.5, 0.8)
            pairs.append((src, dst, w, mf))
        bundles, ecaps, mcaps = generate_benchmark_instance(base, pairs, rng)
        instances[f"n{n_req}"] = {
            "bundles": bundles,
            "edge_capacities": ecaps,
            "memory_capacities": mcaps,
            "n_requests": n_req,
            "request_pairs": [(s, d) for s, d, _, _ in pairs],
        }
    return instances
