"""Extension 9 -- multi-objective allocation and Pareto frontiers.

The scalar utility in the paper collapses everything into one number.  A
network operator instead cares about several objectives at once:

    max [ delivered fidelity, success probability, throughput ]
    min [ latency, memory footprint, Bell-pair consumption ].

This module treats the bundle-allocation problem as a multi-objective
optimization.  For small instances (``product |strategies_i| <= max_combos``)
the *exact* Pareto frontier is enumerated and reported; for larger instances a
weighted-sum scalarization delegates back to the Metropolis annealer.  It also
implements the practical *constraint queries*:

    * "highest throughput while keeping mean fidelity >= F_target"
    * "minimum memory while keeping acceptance >= A_target"

whose answers the plot layer turns into frontier curves.
"""

from collections import defaultdict
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

# objective keys and their optimization directions
MAXIMIZE = ("fidelity", "success", "throughput")
MINIMIZE = ("latency", "memory", "bell_pairs")


def selection_objectives(bundles: List[dict],
                         selected: List[Tuple[str, str]]) -> Dict[str, float]:
    """Objective vector of a selection.

    ``fidelity``/``success`` are the mean over served requests; ``latency``,
    ``memory`` and ``bell_pairs`` are totals; ``throughput`` is the number of
    requests served.
    """
    util_of = {}
    fids, succs, lats, bells = [], [], [], []
    mem_total = 0.0
    for b in bundles:
        util_of[(b["request_id"], b["bundle_id"])] = b
    for rid, bid in selected:
        b = util_of.get((rid, bid))
        if b is None:
            continue
        fids.append(b.get("fidelity", 0.0))
        succs.append(b.get("success_probability", 1.0))
        lats.append(b.get("latency", 0.0))
        bells.append(b.get("bell_pair_cost", 1))
        mem_total += sum(b.get("memory_demands", {}).values())
    return {
        "throughput": len(fids),
        "fidelity": sum(fids) / len(fids) if fids else 0.0,
        "success": sum(succs) / len(succs) if succs else 0.0,
        "latency": sum(lats),
        "memory": mem_total,
        "bell_pairs": sum(bells),
    }


def _feasible(selections: Dict[str, str], bundles: List[dict],
              edge_capacities: dict, memory_capacities: dict) -> bool:
    edge_load: Dict[tuple, int] = defaultdict(int)
    mem_load: Dict[str, int] = defaultdict(int)
    for b in bundles:
        rid, bid = b["request_id"], b["bundle_id"]
        if selections.get(rid) != bid:
            continue
        for e, d in b["edge_demands"].items():
            edge_load[tuple(sorted(e))] += d
        for n, d in b["memory_demands"].items():
            mem_load[n] += d
    for e, load in edge_load.items():
        if load > edge_capacities.get(e, 10 ** 18):
            return False
    for n, load in mem_load.items():
        if load > memory_capacities.get(n, 10 ** 18):
            return False
    return True


def _selections_iter(bundles: List[dict], max_combos: int = 20000):
    """Brute-force enumeration of every per-request selection.

    Each request may pick any of its bundles *or be dropped* (``None``), which
    mirrors the paper's optimizers where an unschedulable request is left
    unserved.  ``max_combos`` guards the product of strategy-set sizes.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for b in bundles:
        groups[b["request_id"]].append(b)
    orders = [[(r, None)] + [(r, bl["bundle_id"]) for bl in groups[r]]
              for r in groups]
    n = 1
    for bs in orders:
        n *= len(bs)
    if n > max_combos:
        raise ValueError(f"{n} combinations exceeds max_combos={max_combos}")
    for combo in product(*orders):
        yield dict(combo)


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """Strict Pareto dominance: a >= b on every objective, > on at least one
    (MAXIMIZE objectives are maximized, MINIMIZE minimized)."""
    at_least = True
    strictly = False
    for k in MAXIMIZE:
        if a[k] < b[k] - 1e-12:
            at_least = False
    for k in MINIMIZE:
        if a[k] > b[k] + 1e-12:
            at_least = False
    for k in MAXIMIZE:
        if a[k] > b[k] + 1e-12:
            strictly = True
    for k in MINIMIZE:
        if a[k] < b[k] - 1e-12:
            strictly = True
    return at_least and strictly


def pareto_frontier(bundles: List[dict], edge_capacities: dict,
                    memory_capacities: dict, max_combos: int = 20000,
                    keys: Optional[List[str]] = None) -> List[Dict]:
    """Exact Pareto-optimal selection set (returns {selection, objectives})."""
    if keys is None:
        keys = list(MAXIMIZE) + list(MINIMIZE)
    points = []
    for selections in _selections_iter(bundles, max_combos=max_combos):
        if not _feasible(selections, bundles, edge_capacities, memory_capacities):
            continue
        sel_list = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        objectives = selection_objectives(bundles, sel_list)
        points.append({"selection": sel_list,
                       "objectives": {k: objectives[k] for k in keys}})

    non_dominated = []
    for p in points:
        dominated = False
        for q in points:
            if p is q:
                continue
            if _dominates(q["objectives"], p["objectives"]):
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)
    return non_dominated


def weighted_score(objectives: Dict[str, float], weights: Dict[str, float]) -> float:
    """Scalarized objective: sum w_k * (normalized direction) * value."""
    total = 0.0
    for k, w in weights.items():
        if w == 0.0:
            continue
        if k in MAXIMIZE:
            total += w * objectives.get(k, 0.0)
        elif k in MINIMIZE:
            total -= w * objectives.get(k, 0.0)
        else:
            raise ValueError(f"unknown objective {k!r}")
    return total


def solve_weighted(bundles: List[dict], edge_capacities: dict,
                   memory_capacities: dict, weights: Dict[str, float],
                   max_combos: int = 20000) -> Dict[str, Optional[str]]:
    """Best selection under a weighted-sum scalarization (exact enumerate)."""
    best = None
    best_score = -float("inf")
    for selections in _selections_iter(bundles, max_combos=max_combos):
        if not _feasible(selections, bundles, edge_capacities, memory_capacities):
            continue
        sel_list = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        score = weighted_score(selection_objectives(bundles, sel_list), weights)
        if score > best_score:
            best_score = score
            best = selections
    return best


def constraint_frontier(bundles: List[dict], edge_capacities: dict,
                        memory_capacities: dict, targets: List[float],
                        constrain: str = "fidelity",
                        maximize: str = "throughput",
                        max_combos: int = 20000) -> List[dict]:
    """Epsilon-constraint curve: maximize ``maximize`` subject to
    ``constrain >= target`` for each target in ``targets``.

    Returns rows with ``target``, the achieved constraint value, the
    maximized objective, and the full objective vector of the best selection.
    """
    rows = []
    for target in targets:
        best = None
        best_value = -float("inf")
        for selections in _selections_iter(bundles, max_combos=max_combos):
            if not _feasible(selections, bundles, edge_capacities, memory_capacities):
                continue
            sel_list = [(rid, bid) for rid, bid in selections.items() if bid is not None]
            objs = selection_objectives(bundles, sel_list)
            if objs[constrain] < target - 1e-12:
                continue
            value = objs[maximize]
            if value > best_value:
                best_value = value
                best = (sel_list, objs)
        if best is None:
            rows.append({"target": target, constrain: None,
                         maximize: None, "feasible": False})
            continue
        sel_list, objs = best
        rows.append({"target": target,
                     **{k: objs[k] for k in objs},
                     "feasible": True,
                     "selection": sel_list})
    return rows
