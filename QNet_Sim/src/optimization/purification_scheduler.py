"""Purification as a first-class optimization variable (Wave 3, item 2).

The bundle model already enumerates purification levels implicitly (``q``
rounds per link), but the *decision* dimension is usually hidden inside a
single ``purification_rounds`` field and dominated bundles are pruned away
before the optimizer ever sees them.  This module promotes purification to an
explicit, inspectable decision variable:

* ``purification_tradeoff`` -- for a fixed path, the full
  ``(k, F(k), P(k), T(k), cost(k), memory(k))`` family across ``k``
  purification rounds.  Purification raises end-to-end fidelity but
  *quadratically* raises raw Bell-pair cost ``2**k``, latency, memory demand
  and *lowers* success probability, so the utility ``u(k)`` is non-monotone:
  this is exactly the fidelity <-> memory <-> latency <-> success trade-off
  that makes purification an interesting optimization variable.
* ``purification_frontier`` -- the Pareto frontier over
  ``(fidelity, success_probability, latency, bell_pair_cost, memory)``.
* ``run_purification_comparison`` -- purification-agnostic (``k in {0}``),
  default (``k in {0,1,2}``) and purification-aware (``k in {0..K}``)
  optimization on identical instances, reporting the *purification profile*
  the optimizer actually picks (mean rounds, mean raw-pair cost, mean memory,
  mean success probability) alongside utility.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from experiments.instances import (
    _evaluate_bundle, _shortest_paths, generate_chain_topology,
)
from routing.utility import UtilityModel


def purification_tradeoff(topology: dict, path: List[str],
                         min_fidelity: float, weight: float = 1.0,
                         q_max: int = 4) -> List[dict]:
    """Score every purification level ``k = 0..q_max`` on one path.

    Returns one dict per level (fidelity, success probability, latency,
    Bell-pair cost, memory demand, utility), or an empty list if the path
    cannot meet ``min_fidelity`` at any level.
    """
    rows = []
    for k in range(q_max + 1):
        b = _evaluate_bundle(topology, path, k, path[0], path[-1],
                             weight, min_fidelity)
        if b is None:
            continue
        rows.append({
            "k": k,
            "fidelity": b["fidelity"],
            "success_probability": b["success_probability"],
            "latency": b["latency"],
            "bell_pair_cost": b["bell_pair_cost"],
            "memory_demand": sum(b["memory_demands"].values()),
            "utility": b["utility"],
        })
    return rows


def purification_frontier(topology: dict, path: List[str],
                          min_fidelity: float, weight: float = 1.0,
                          q_max: int = 4) -> List[dict]:
    """Pareto frontier over fidelity / success / latency / cost / memory.

    A purification level ``k`` dominates ``k'`` if it is at least as good on
    every objective and strictly better on at least one.  The frontier exposes
    the levels a scheduler should actually consider.
    """
    rows = purification_tradeoff(topology, path, min_fidelity, weight, q_max)
    non_dominated = []
    for p in rows:
        dominated = False
        for q in rows:
            if q["k"] == p["k"]:
                continue
            better = (q["fidelity"] >= p["fidelity"] - 1e-12
                      and q["success_probability"] >= p["success_probability"] - 1e-12
                      and q["latency"] <= p["latency"] + 1e-12
                      and q["bell_pair_cost"] <= p["bell_pair_cost"] + 1e-12
                      and q["memory_demand"] <= p["memory_demand"] + 1e-12)
            strict = (q["fidelity"] > p["fidelity"] + 1e-12
                      or q["success_probability"] > p["success_probability"] + 1e-12
                      or q["latency"] < p["latency"] - 1e-12
                      or q["bell_pair_cost"] < p["bell_pair_cost"] - 1e-12
                      or q["memory_demand"] < p["memory_demand"] - 1e-12)
            if better and strict:
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)
    return non_dominated


def _build_instance(topology: dict, n_requests: int, seed: int,
                    q_values: Optional[List[int]] = None,
                    fidelity_bounds: Tuple[float, float] = (0.5, 0.7)) -> List[dict]:
    from experiments.instances import generate_request_bundles
    rng = random.Random(seed)
    nodes = topology["nodes"]
    out = []
    for i in range(n_requests):
        src, dst = rng.sample(nodes, 2)
        w = rng.uniform(10.0, 100.0)
        mf = rng.uniform(*fidelity_bounds)
        bs = generate_request_bundles(topology, src, dst, w, mf,
                                      q_values=q_values)
        for b in bs:
            b["request_id"] = f"req_{i}"
            b["bundle_id"] = f"req_{i}_p{bs.index(b)}"
            b["min_fidelity"] = mf
            b["weight"] = w
        out.extend(bs)
    return out


def _selection_metrics(bundles: List[dict],
                       selected: List[Tuple[str, str]]) -> Dict[str, float]:
    util = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    sel = [util[k] for k in selected if k in util]
    k_vals = []
    cost = 0.0
    mem = 0.0
    p_succ = 0.0
    for b in sel:
        k_vals.append(_purification_rounds(b))
        cost += b.get("bell_pair_cost", 0)
        mem += sum(b.get("memory_demands", {}).values())
        p_succ += b.get("success_probability", 0.0)
    n = max(len(sel), 1)
    return {
        "n_served": len(sel),
        "utility": sum(b["utility"] for b in sel),
        "mean_k": sum(k_vals) / n if k_vals else 0.0,
        "mean_bell_pairs": cost / n,
        "mean_memory": mem / n,
        "mean_success": p_succ / n,
    }


def _purification_rounds(bundle: dict) -> int:
    v = bundle.get("purification_rounds")
    if v is not None:
        return int(v)
    import re
    m = re.search(r"_q(\d+)$", bundle.get("bundle_id", ""))
    return int(m.group(1)) if m else 0


def run_purification_comparison(topology: dict, n_requests: int = 8,
                                q_max: int = 4, seed: int = 42) -> Dict:
    """Purification-agnostic vs default vs purification-aware optimization.

    Every regime solves the *same* request pairs with the Metropolis annealer;
    they differ only in the purification levels offered to the optimizer
    (``{0}``, ``{0,1,2}``, ``{0..q_max}``).  Reports utility, served count and
    the purification profile the optimizer selected.
    """
    from optimization.metropolis_annealer import MetropolisAnnealer

    rng = random.Random(seed)
    nodes = topology["nodes"]
    pairs = []
    for _ in range(n_requests):
        src, dst = rng.sample(nodes, 2)
        pairs.append((src, dst, rng.uniform(10.0, 100.0),
                      rng.uniform(0.5, 0.7)))

    ec, mc = topology["edge_capacities"], topology["memory_capacities"]
    rows = []
    regimes = [("purification_agnostic", [0]),
               ("default_q012", [0, 1, 2]),
               ("purification_aware", list(range(q_max + 1)))]
    for label, q_values in regimes:
        bundles = _build_instance(topology, n_requests, seed, q_values=q_values)
        opt = MetropolisAnnealer(bundles, ec, mc, seed=seed)
        t0 = time.perf_counter()
        r = opt.solve(
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        elapsed = time.perf_counter() - t0
        m = _selection_metrics(bundles, r.get("selected", []))
        rows.append({
            "regime": label,
            "n_requests": n_requests,
            "n_bundles": len(bundles),
            "q_max": q_max,
            "utility": m["utility"],
            "served": m["n_served"],
            "served_ratio": m["n_served"] / max(n_requests, 1),
            "mean_k": m["mean_k"],
            "mean_bell_pairs": m["mean_bell_pairs"],
            "mean_memory": m["mean_memory"],
            "mean_success": m["mean_success"],
            "time_s": elapsed,
        })
    return {"n_requests": n_requests, "rows": rows}


def run_purification_sweep(path_lengths: Optional[List[int]] = None,
                           q_max: int = 4,
                           min_fidelity_values: Optional[List[float]] = None,
                           raw_fidelity: float = 0.85,
                           generation_prob: float = 0.8) -> List[dict]:
    """Sweep the purification frontier across path lengths and SLA thresholds.

    Deterministic (no randomness): one fixed chain per path length, all
    purification levels scored.  Shows how the ``k`` that maximizes utility
    shifts with path length and ``F_min``, and where purification *stops
    paying* (higher ``k`` raises cost/latency/memory while lowering success).
    """
    if path_lengths is None:
        path_lengths = [2, 4, 6, 8]
    if min_fidelity_values is None:
        min_fidelity_values = [0.55, 0.6, 0.7, 0.8]
    rows = []
    for n_links in path_lengths:
        n_nodes = n_links + 1
        topo = generate_chain_topology(n_nodes=n_nodes,
                                       edge_capacity=6, memory_capacity=10,
                                       raw_fidelity=raw_fidelity,
                                       generation_prob=generation_prob,
                                       latency=5.0)
        path = topo["nodes"]
        for mf in min_fidelity_values:
            for p in purification_tradeoff(topo, path, mf, weight=50.0,
                                           q_max=q_max):
                rows.append({
                    "path_length": n_links,
                    "min_fidelity": mf,
                    "k": p["k"],
                    "fidelity": p["fidelity"],
                    "success_probability": p["success_probability"],
                    "latency": p["latency"],
                    "bell_pair_cost": p["bell_pair_cost"],
                    "memory_demand": p["memory_demand"],
                    "utility": p["utility"],
                })
    return rows
