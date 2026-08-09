"""Extension 17 -- entanglement-swapping strategy optimization.

The bundle evaluator composes link fidelities left-to-right, i.e. it always
swaps in one fixed order.  A path A-B-C-D-E with links AB, BC, CD, DE can be
swapped in any order --- every full binary tree over the links:

    ((AB)-(BC))-(CD)      vs      (AB)-((BC)-(CD))      vs   ...

Key physical fact (honest, and worth stating in the paper): for Werner
states the swapping map ``swap(a, b) = ab + (1-a)(1-b)/3`` is *associative*,
so every tree gives the *same* noiseless end-to-end fidelity.  The ordering
still matters for the resources around it:

* **timing / coherence** -- a tree's depth is the number of sequential BSM
  rounds; the deepest held pair waits ``depth * delta`` before consumption,
  and under T1/T2 storage decay the delivered fidelity falls with that wait,
  so shallower trees (balanced / parallel) preserve more fidelity,
* **memory concurrency** -- a shallow tree runs more swaps in parallel, so
  the peak number of simultaneously held pairs is larger.  There is a
  genuine trade-off frontier: depth (coherence) vs peak concurrency (memory).

This module enumerates the full tree set (Catalan-many) for a path, scores
each tree by ``(depth, peak_concurrency, delivered_fidelity)``, exposes the
canonical strategies (``linear``, ``reverse``, ``balanced``, ``optimal``),
and reports the depth/concurrency/fidelity frontier.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

from fidelity.fidelity_model import FidelityModel


def swap_fidelity(f1: float, f2: float) -> float:
    """End-to-end fidelity after one swapping operation (Werner states)."""
    return FidelityModel.entanglement_swapping(f1, f2)


def _decay_relative(hold_time: float, tau_mem: float,
                    t1: Optional[float] = None) -> float:
    """T1/T2 storage-decay multiplier for a pair held ``hold_time``."""
    if hold_time <= 0 or tau_mem <= 0:
        return 1.0
    t2 = tau_mem
    t1v = t1 if t1 and t1 > 0 else 2.0 * t2
    return 0.25 * (1.0 + 3.0 * math.exp(-hold_time / t2)
                   * (1.0 + math.exp(-hold_time / t1v)) / 2.0)


# ----------------------------------------------------------------------
# swapping trees (full binary trees over the links)
# ----------------------------------------------------------------------
# A tree is either ``("leaf",)`` or ``("node", left, right)``.  Every full
# binary tree over n leaves is one valid swapping order (n - 1 BSMs); the
# number of trees is the Catalan number C_{n-1}.

def _trees(n_leaves: int):
    if n_leaves == 1:
        return [("leaf",)]
    out = []
    for k in range(1, n_leaves):
        for left in _trees(k):
            for right in _trees(n_leaves - k):
                out.append(("node", left, right))
    return out


def count_leaves(tree) -> int:
    if tree[0] == "leaf":
        return 1
    return count_leaves(tree[1]) + count_leaves(tree[2])


def tree_depth(tree) -> int:
    """Number of sequential BSM rounds on the longest root-to-leaf path."""
    if tree[0] == "leaf":
        return 0
    return 1 + max(tree_depth(tree[1]), tree_depth(tree[2]))


def tree_peak_concurrency(tree) -> int:
    """Maximum number of swaps running in the same round (memory footprint).

    A swap at an internal node can start only after both of its children have
    been swapped, so its round is ``1 + max(round(left), round(right))`` (with
    leaves at round 0).  The peak concurrency is the maximum number of swaps
    scheduled in the same round.
    """
    rounds: Dict[int, int] = {}

    def _rounds(t):
        if t[0] == "leaf":
            return 0
        r = 1 + max(_rounds(t[1]), _rounds(t[2]))
        rounds[r] = rounds.get(r, 0) + 1
        return r

    _rounds(tree)
    return max(rounds.values(), default=0)


def evaluate_tree(tree, fids: List[float], start: int = 0) -> float:
    """Noiseless end-to-end fidelity of a swapping tree over ``fids``."""
    if tree[0] == "leaf":
        return fids[start]
    f1 = evaluate_tree(tree[1], fids, start)
    f2 = evaluate_tree(tree[2], fids, start + count_leaves(tree[1]))
    return swap_fidelity(f1, f2)


def linear_tree(n_leaves: int):
    """Left-to-right (or right-to-left) sequential swapping."""
    if n_leaves == 1:
        return ("leaf",)
    return ("node", linear_tree(n_leaves - 1), ("leaf",))


def balanced_tree(n_leaves: int):
    """Balanced tree: minimum depth, maximally parallel BSMs."""
    if n_leaves == 1:
        return ("leaf",)
    left_n = n_leaves // 2
    return ("node", balanced_tree(left_n), balanced_tree(n_leaves - left_n))


# ----------------------------------------------------------------------
# strategy evaluation
# ----------------------------------------------------------------------
def linear_fidelity(fids: List[float]) -> float:
    """Noiseless fidelity under the paper's default (linear) order."""
    if not fids:
        return 0.0
    f = fids[0]
    for nxt in fids[1:]:
        f = swap_fidelity(f, nxt)
    return f


def strategy_metrics(tree, fids: List[float], delta: float = 1.0,
                     tau_mem: float = float("inf"),
                     t1: Optional[float] = None) -> Dict[str, float]:
    """Depth, peak concurrency and delivered fidelity of one strategy tree.

    ``delta`` is the time per BSM round; the deepest pair is held for
    ``depth * delta`` before consumption.  ``delivered_fidelity`` combines the
    (order-independent) noiseless swapped fidelity with the storage decay.
    """
    depth = tree_depth(tree)
    concurrency = max(1, tree_peak_concurrency(tree))
    noiseless = evaluate_tree(tree, fids)
    hold = depth * delta
    delivered = noiseless * _decay_relative(hold, tau_mem, t1)
    return {
        "depth": depth,
        "peak_concurrency": concurrency,
        "fidelity": noiseless,
        "delivered_fidelity": delivered,
        "hold_time": hold,
    }


def all_strategies(fids: List[float], delta: float = 1.0,
                   tau_mem: float = float("inf"),
                   t1: Optional[float] = None) -> List[Dict]:
    """Every swapping tree scored; also tagged with its strategy family."""
    trees = _trees(len(fids))
    families = {}
    for tree in trees:
        if tree == linear_tree(len(fids)):
            families[tuple(tree)] = "linear"
        elif tree == balanced_tree(len(fids)):
            families[tuple(tree)] = "balanced"
    out = []
    for tree in trees:
        m = strategy_metrics(tree, fids, delta, tau_mem, t1)
        out.append({"tree": tree, "strategy": families.get(tuple(tree), "other"),
                    **m})
    return out


def strategy_fidelity(fids: List[float], strategy: str = "linear",
                      delta: float = 1.0, tau_mem: float = float("inf"),
                      t1: Optional[float] = None) -> Dict[str, float]:
    """Delivered fidelity for one of ``linear`` / ``reverse`` / ``balanced``
    / ``optimal`` (the minimum-depth tree)."""
    n = len(fids)
    if strategy == "linear":
        tree = linear_tree(n)
    elif strategy == "reverse":
        tree = linear_tree(n)  # reverse swaps are the same tree, mirrored
    elif strategy == "balanced":
        tree = balanced_tree(n)
    elif strategy == "optimal":
        candidates = _trees(n)
        tree = min(candidates, key=lambda t: (tree_depth(t),
                                              -tree_peak_concurrency(t)))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return strategy_metrics(tree, fids, delta, tau_mem, t1)


def run_swapping_order_sweep(path_lengths: Optional[List[int]] = None,
                             n_trials: int = 60, seed: int = 42,
                             delta: float = 1.0,
                             tau_mem: Optional[float] = None) -> List[dict]:
    """Compare the canonical strategies under coherence decay.

    Returns one row per (path length, trial, strategy): noiseless and
    delivered fidelity, hold time and peak concurrency.  The story the data
    tells: every strategy has the same *noiseless* fidelity (Werner swap is
    associative), but under T1/T2 decay the shallower (balanced) trees win on
    delivered fidelity while the sequential (linear) tree uses least memory.
    """
    if path_lengths is None:
        path_lengths = [3, 4, 5, 6, 7, 8]
    if tau_mem is None:
        tau_mem = 5.0 * delta
    rng = random.Random(seed)
    rows = []
    for n_links in path_lengths:
        for _ in range(n_trials):
            fids = [rng.uniform(0.6, 0.95) for _ in range(n_links)]
            for strategy in ["linear", "balanced", "optimal"]:
                m = strategy_fidelity(fids, strategy, delta=delta,
                                      tau_mem=tau_mem)
                rows.append({
                    "path_length": n_links,
                    "strategy": strategy,
                    "n_links": n_links,
                    **m,
                    "fid_loss": m["fidelity"] - m["delivered_fidelity"],
                })
    return rows


def run_path_fidelity_sweep(path_length: int = 10, link_fidelity: float = 0.85,
                            delta: float = 1.0,
                            tau_mem_values: Optional[List[float]] = None) -> List[dict]:
    """Deterministic: one fixed path, delivered fidelity vs coherence lifetime.

    This isolates the swapping-strategy effect on a single long path.  Every
    strategy delivers the same noiseless fidelity; as ``tau_mem`` shrinks
    relative to the path's BSM depth, the balanced / optimal (minimum-depth)
    strategies preserve dramatically more fidelity than the linear order.
    """
    if tau_mem_values is None:
        tau_mem_values = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    fids = [link_fidelity] * path_length
    rows = []
    for tau in tau_mem_values:
        for strategy in ["linear", "balanced", "optimal"]:
            m = strategy_fidelity(fids, strategy, delta=delta, tau_mem=tau)
            rows.append({
                "path_length": path_length,
                "strategy": strategy,
                "tau_mem": tau,
                **m,
            })
    return rows


def strategy_frontier(fids: List[float], delta: float = 1.0,
                      tau_mem: float = float("inf"),
                      t1: Optional[float] = None) -> List[Dict]:
    """Pareto frontier over (depth, peak_concurrency, delivered_fidelity).

    Returns the non-dominated strategy trees, exposing the depth-vs-memory
    trade-off a scheduler faces when choosing a swapping order.
    """
    scored = all_strategies(fids, delta, tau_mem, t1)
    non_dominated = []
    for i, p in enumerate(scored):
        dominated = False
        for j, q in enumerate(scored):
            if i == j:
                continue
            better = (q["depth"] <= p["depth"]
                      and q["peak_concurrency"] <= p["peak_concurrency"]
                      and q["delivered_fidelity"] >= p["delivered_fidelity"] - 1e-12)
            strict = (q["depth"] < p["depth"]
                      or q["peak_concurrency"] < p["peak_concurrency"]
                      or q["delivered_fidelity"] > p["delivered_fidelity"] + 1e-12)
            if better and strict:
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)
    return non_dominated


# ----------------------------------------------------------------------
# swapping-order-aware bundle generation
# ----------------------------------------------------------------------
def optimal_order_bundle(topology: dict, path: List[str], q: int,
                         source: str, target: str, weight: float,
                         min_fidelity: float, tau_mem: float = 5.0,
                         delta: float = 1.0) -> Optional[dict]:
    """Re-evaluate a request bundle under the best swapping strategy.

    Recovers the per-link post-purification fidelities, evaluates every
    swapping tree, and picks the tree maximising the *delivered* fidelity
    (noiseless fidelity degraded by the max memory hold time under T1/T2
    decay).  Returns a bundle carrying the winning tree's fidelity, latency
    and utility, or None if no order reaches ``min_fidelity``.
    """
    from experiments.instances import _evaluate_bundle
    base = _evaluate_bundle(topology, path, q, source, target, weight,
                            min_fidelity)
    if base is None:
        return None
    link_fids = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = (u, v) if u < v else (v, u)
        lp = topology["link_params"].get(edge,
                                         topology["link_params"].get((v, u)))
        if lp is None:
            return None
        f = lp["raw_fidelity"]
        for _ in range(q):
            if f <= 0.5:
                break
            f = FidelityModel.purification_bbpssw(f)
        link_fids.append(f)

    best = None
    for tree in _trees(len(link_fids)):
        m = strategy_metrics(tree, link_fids, delta, tau_mem)
        if best is None or m["delivered_fidelity"] > best["delivered_fidelity"]:
            best = m
    best_f = best["delivered_fidelity"]
    if best_f < min_fidelity:
        return None

    margin = max(0.0, best_f - min_fidelity)
    latency = base["latency"] + best["depth"] * delta
    utility = (weight * base["success_probability"] * (1.0 + margin)
               - 0.01 * latency - 0.05 * base["bell_pair_cost"])
    out = dict(base)
    out["fidelity"] = best_f
    out["utility"] = utility
    out["swapping_strategy"] = "optimal"
    out["swapping_depth"] = best["depth"]
    out["latency"] = latency
    return out


def _delivered_fidelity(bundle: dict, tau_mem: float, delta: float) -> float:
    """Delivered fidelity of a selected bundle under the storage-decay model.

    ``optimal_order_bundle`` stores the delivered fidelity directly; a linear
    (default) bundle stores the noiseless fidelity, so we degrade it by the
    linear strategy's maximum hold time ``(n_links - 1) * delta``.
    """
    if "swapping_strategy" in bundle:
        return bundle["fidelity"]
    n_links = max(0, len(bundle.get("path", [])) - 1)
    hold = max(0, n_links - 1) * delta
    return bundle.get("fidelity", 0.0) * _decay_relative(hold, tau_mem)


def run_swapping_bundle_comparison(topology: dict, n_requests: int = 8,
                                   tau_mem: float = 3.0, delta: float = 1.0,
                                   seed: int = 42) -> Dict:
    """Utility / delivered-fidelity gain from swapping-order optimization.

    Both the default (linear-order) bundle set and the optimal-order bundle
    set are solved by the Metropolis annealer; the aggregate utility and mean
    *delivered* fidelity (after memory storage decay) differences measure the
    value of choosing the swapping strategy.
    """
    import time
    import random as _random
    from optimization.metropolis_annealer import MetropolisAnnealer

    rng = _random.Random(seed)
    nodes = topology["nodes"]
    pairs = []
    for _ in range(n_requests):
        src, dst = rng.sample(nodes, 2)
        pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))

    def _bundles(optimal: bool):
        from experiments.instances import _shortest_paths, _pareto_prune
        from experiments.instances import _evaluate_bundle
        out = []
        for i, (src, dst, w, mf) in enumerate(pairs):
            paths = _shortest_paths(topology, src, dst, k=3)
            for pi, path in enumerate(paths):
                for q in [0, 1, 2]:
                    if optimal:
                        b = optimal_order_bundle(topology, path, q, src, dst,
                                                 w, mf, tau_mem=tau_mem,
                                                 delta=delta)
                    else:
                        b = _evaluate_bundle(topology, path, q, src, dst, w, mf)
                    if b is not None:
                        b["request_id"] = f"req_{i}"
                        b["bundle_id"] = f"req_{i}_p{pi}_q{q}"
                        out.append(b)
        return _pareto_prune(out)

    ec, mc = topology["edge_capacities"], topology["memory_capacities"]
    rows = []
    for label, optimal in [("linear", False), ("optimal_order", True)]:
        bundles = _bundles(optimal)
        util_of = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
        opt = MetropolisAnnealer(bundles, ec, mc, seed=seed)
        t0 = time.perf_counter()
        r = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        elapsed = time.perf_counter() - t0
        sel = r.get("selected", [])
        delivered = []
        for rid, bid in sel:
            for b in bundles:
                if b["request_id"] == rid and b["bundle_id"] == bid:
                    delivered.append(_delivered_fidelity(b, tau_mem, delta))
                    break
        rows.append({
            "swapping_strategy": label,
            "n_requests": n_requests,
            "n_bundles": len(bundles),
            "served": len(sel),
            "served_ratio": len(sel) / max(n_requests, 1),
            "utility": sum(util_of.get(k, 0.0) for k in sel),
            "mean_delivered_fidelity": (sum(delivered) / len(delivered)
                                        if delivered else 0.0),
            "time_s": elapsed,
        })
    return {"n_requests": n_requests, "rows": rows}
