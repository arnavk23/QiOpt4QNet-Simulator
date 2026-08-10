"""Extension 11 -- k-disjoint entanglement path provisioning.

Single-path routing puts all of a request's Bell pairs on one route.  When a
request asks for multiple end-to-end pairs (or the network is congested), it
can instead provision *several edge-disjoint paths at once*:

    A -> B :  P_1, P_2, ..., P_k     (edge-disjoint)

Multi-entanglement routing increases the offered capacity, adds redundancy
(the request succeeds if *any* of its independent paths succeeds) and relieves
congestion by spreading demand.  This module:

* ``k_disjoint_paths`` -- Suurballe-style edge-disjoint paths (networkx),
* ``generate_multipath_bundles`` -- builds, per request, ordinary single-path
  bundles *plus* composite bundles that reserve every non-empty subset of the
  disjoint paths.  Composites keep the same ``{bundle_id, edge_demands,
  memory_demands, utility}`` optimizer schema, so *no solver changes* are
  needed: the optimizer itself decides single- vs multi-path provisioning,

* ``run_disjoint_comparison`` -- benchmarks a capacity-constrained network
  with single-path-only bundles vs the full (multipath-enabled) candidate set,
  reporting served ratio / utility / redundancy gained.

The composite bundle model: provisioning a subset S of edge-disjoint paths
gives the request an effective success probability
``1 - prod_{p in S} (1 - p_p)`` (independent attempts), the fidelity of the
best prepared pair ``max f_p``, and a latency of ``max lat_p`` (paths are
prepared in parallel).  Resource demands and Bell-pair costs sum across paths.
"""

from collections import defaultdict
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx


def k_disjoint_paths(topology: dict, source: str, target: str,
                     k: int = 3) -> List[List[str]]:
    """Up to ``k`` edge-disjoint paths (networkx edge_disjoint_paths)."""
    G = nx.Graph()
    G.add_nodes_from(topology["nodes"])
    G.add_edges_from(topology["edges"])
    paths = []
    try:
        for path in nx.edge_disjoint_paths(G, source, target):
            paths.append(list(path))
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return paths


def _evaluate_single(topology: dict, path: List[str], q: int,
                     source: str, target: str, weight: float,
                     min_fidelity: float) -> Optional[dict]:
    """Reuse the paper's per-path bundle evaluator for one path."""
    from experiments.instances import _evaluate_bundle
    return _evaluate_bundle(topology, path, q, source, target, weight,
                            min_fidelity)


def _combine_multipath(singles: List[dict], source: str, target: str,
                       weight: float, min_fidelity: float,
                       bundle_id: str) -> dict:
    """Combine single-path bundles (on disjoint routes) into one bundle."""
    p_succ = 1.0
    fidelity = 0.0
    latency = 0.0
    bell_cost = 0
    edge_demands: Dict[tuple, int] = defaultdict(int)
    mem_demands: Dict[str, int] = defaultdict(int)
    for b in singles:
        p_succ *= 1.0 - b["success_probability"]
        fidelity = max(fidelity, b["fidelity"])
        latency = max(latency, b.get("latency", 0.0))
        bell_cost += b.get("bell_pair_cost", 1)
        for e, d in b["edge_demands"].items():
            edge_demands[tuple(sorted(e))] += d
        for n, d in b["memory_demands"].items():
            mem_demands[n] += d
    success = 1.0 - p_succ
    margin = max(0.0, fidelity - min_fidelity)
    utility = weight * success * (1.0 + margin) - 0.01 * latency - 0.05 * bell_cost
    return {
        "bundle_id": bundle_id,
        "request_id": f"req_{source}_{target}",
        "path": [b["path"] for b in singles],
        "edge_demands": dict(edge_demands),
        "memory_demands": dict(mem_demands),
        "utility": utility,
        "latency": latency,
        "fidelity": fidelity,
        "success_probability": success,
        "bell_pair_cost": bell_cost,
        "n_paths": len(singles),
        "paths": [b["path"] for b in singles],
    }


def generate_multipath_bundles(topology: dict, source: str, target: str,
                               weight: float, min_fidelity: float,
                               k_paths: int = 3,
                               q_values: Optional[List[int]] = None,
                               include_singles: bool = True,
                               include_multi: bool = True) -> List[dict]:
    """Single-path + composite multipath bundles for one request.

    Every non-empty subset of the k edge-disjoint paths becomes a candidate
    bundle (sized 1 for singles, 2..k for composites).  The bundle set is
    Pareto-pruned on (fidelity, cost, latency) to drop dominated candidates.
    """
    if q_values is None:
        q_values = [0, 1, 2]
    paths = k_disjoint_paths(topology, source, target, k=k_paths)
    if len(paths) < 2:
        # No disjoint alternatives: fall back to plain single-path bundles.
        from experiments.instances import generate_request_bundles
        return generate_request_bundles(topology, source, target, weight,
                                        min_fidelity, q_values=q_values)

    by_path: Dict[int, List[dict]] = {}
    for pi, path in enumerate(paths):
        by_path[pi] = []
        for q in q_values:
            b = _evaluate_single(topology, path, q, source, target, weight,
                                 min_fidelity)
            if b is not None:
                by_path[pi].append(b)

    candidates: List[dict] = []
    rid = f"req_{source}_{target}"
    if include_singles:
        for pi, bs in by_path.items():
            for j, b in enumerate(bs):
                b["request_id"] = rid
                b["bundle_id"] = f"{rid}_p{pi}_q{j}"
                candidates.append(b)
    if include_multi:
        for size in range(2, len(paths) + 1):
            for combo_idx, combo in enumerate(combinations(range(len(paths)), size)):
                valid = True
                for pi in combo:
                    if not by_path[pi]:
                        valid = False
                        break
                if not valid:
                    continue
                # take the best single bundle per path (highest utility)
                singles = [max(by_path[pi], key=lambda b: b["utility"])
                           for pi in combo]
                combined = _combine_multipath(singles, source, target, weight,
                                              min_fidelity,
                                              bundle_id=f"{rid}_m{combo_idx}")
                candidates.append(combined)

    # Pareto-prune on (fidelity, cost, latency) *within* the single-path and
    # multipath groups separately: a composite bundle is not comparable to a
    # single path on cost/latency alone because it trades those for
    # redundancy (higher success probability), which is the whole point.
    from experiments.instances import _pareto_prune
    singles = [b for b in candidates if b.get("n_paths", 1) == 1]
    multis = [b for b in candidates if b.get("n_paths", 1) > 1]
    return _pareto_prune(singles) + _pareto_prune(multis)


def _build_instance_bundles(topology: dict, pairs: List[Tuple[str, str, float, float]],
                            rng, multipath: bool) -> List[dict]:
    all_bundles = []
    for i, (src, dst, weight, min_fid) in enumerate(pairs):
        if multipath:
            bs = generate_multipath_bundles(topology, src, dst, weight,
                                            min_fid, k_paths=3)
        else:
            from experiments.instances import generate_request_bundles
            bs = generate_request_bundles(topology, src, dst, weight, min_fid)
        for j, b in enumerate(bs):
            b["request_id"] = f"req_{i}"
            b["bundle_id"] = f"req_{i}_b{j}"
        all_bundles.extend(bs)
    return all_bundles


def run_disjoint_comparison(topology: dict, n_requests: int = 8,
                            n_expected_disjoint: int = 2,
                            seed: int = 42) -> Dict:
    """Single-path vs multipath provisioning on a capacity-constrained network.

    Returns per-request aggregate metrics for the Metropolis solver on (a)
    single-path-only candidates and (b) the multipath-enabled candidate set,
    plus how often multipath bundles were actually selected.
    """
    import time
    import random as _random
    from optimization.metropolis_annealer import MetropolisAnnealer

    rng = _random.Random(seed)
    nodes = topology["nodes"]
    pairs = []
    attempts = 0
    while len(pairs) < n_requests and attempts < 100:
        attempts += 1
        src, dst = rng.sample(nodes, 2)
        if len(k_disjoint_paths(topology, src, dst, k=3)) >= n_expected_disjoint:
            pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))

    if not pairs:
        raise ValueError("no request pair has enough disjoint paths; "
                         "increase topology connectivity")

    ec, mc = topology["edge_capacities"], topology["memory_capacities"]
    single_b = _build_instance_bundles(topology, pairs, rng, multipath=False)
    multi_b = _build_instance_bundles(topology, pairs, rng, multipath=True)

    util_of = {(b["request_id"], b["bundle_id"]): b["utility"]
               for b in multi_b}

    rows = []
    n_multi_used = 0
    for label, bundles in [("single_path", single_b), ("multipath", multi_b)]:
        opt = MetropolisAnnealer(bundles, ec, mc, seed=seed)
        t0 = time.perf_counter()
        r = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        elapsed = time.perf_counter() - t0
        sel = r.get("selected", [])
        served = len(sel)
        util = sum(util_of.get(k, 0.0) for k in sel)
        n_paths_used = []
        for rid, bid in sel:
            for b in bundles:
                if b["request_id"] == rid and b["bundle_id"] == bid:
                    n_paths_used.append(b.get("n_paths", 1))
                    break
        if label == "multipath":
            n_multi_used = sum(1 for k in sel if n_paths_used and _used_multi(bundles, k))
        rows.append({
            "candidate_set": label,
            "n_requests": n_requests,
            "n_bundles": len(bundles),
            "served": served,
            "served_ratio": served / max(n_requests, 1),
            "utility": util,
            "time_s": elapsed,
            "mean_n_paths": (sum(n_paths_used) / len(n_paths_used)
                             if n_paths_used else 0.0),
        })

    return {
        "n_requests": n_requests,
        "n_bundles_single": len(single_b),
        "n_bundles_multipath": len(multi_b),
        "rows": rows,
        "n_multipath_selected": n_multi_used,
    }


def _used_multi(bundles: List[dict], key: Tuple[str, str]) -> bool:
    for b in bundles:
        if b["request_id"] == key[0] and b["bundle_id"] == key[1]:
            return b.get("n_paths", 1) > 1
    return False
