from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

Edge = Tuple[str, str]


def _undirected(edge: Tuple[str, str]) -> Edge:
    return tuple(sorted(edge))


class FeasibilityChecker:

    def __init__(self, edge_capacities: Dict[Edge, int], memory_capacities: Dict[str, int]):
        self.edge_capacities = {_undirected(e): c for e, c in edge_capacities.items()}
        self.memory_capacities = dict(memory_capacities)

    def check(self, selected: List[Tuple[str, str]], bundles_by_key: Dict[Tuple[str, str], dict]) -> dict:
        edge_load: Dict[Edge, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)

        for key in selected:
            b = bundles_by_key[key]
            for edge, d in b["edge_demands"].items():
                edge_load[_undirected(edge)] += d
            for node, d in b["memory_demands"].items():
                mem_load[node] += d

        edge_violations = {
            e: (load, self.edge_capacities.get(e, 0))
            for e, load in edge_load.items() if load > self.edge_capacities.get(e, 0)
        }
        mem_violations = {
            n: (load, self.memory_capacities.get(n, 0))
            for n, load in mem_load.items() if load > self.memory_capacities.get(n, 0)
        }

        return {
            "feasible": not edge_violations and not mem_violations,
            "edge_violations": edge_violations,
            "memory_violations": mem_violations,
            "edge_load": dict(edge_load),
            "memory_load": dict(mem_load),
        }


def compute_metrics(result: dict, bundles: List[dict], edge_capacities: Dict[Edge, int],
                     memory_capacities: Dict[str, int], all_request_ids: Optional[List[str]] = None,
                     reference_utility: Optional[float] = None) -> dict:
    """
    result: output of allocator.solve(), i.e. {"selected": [(rid, bid), ...], ...}
    Returns a metrics dict directly comparable across any allocator.
    """
    bundles_by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    selected = result["selected"]

    checker = FeasibilityChecker(edge_capacities, memory_capacities)
    feas = checker.check(selected, bundles_by_key)

    total_utility = sum(bundles_by_key[k]["utility"] for k in selected)
    n_requests = len(all_request_ids) if all_request_ids is not None else len({b["request_id"] for b in bundles})
    accepted = len(selected)

    fidelities = [bundles_by_key[k]["fidelity"] for k in selected if "fidelity" in bundles_by_key[k]]
    success_probs = [bundles_by_key[k]["success_probability"] for k in selected if "success_probability" in bundles_by_key[k]]

    edge_caps = {_undirected(e): c for e, c in edge_capacities.items()}
    edge_util = {e: feas["edge_load"].get(e, 0) / c for e, c in edge_caps.items() if c > 0}
    mem_util = {n: feas["memory_load"].get(n, 0) / c for n, c in memory_capacities.items() if c > 0}

    metrics = {
        "method": result.get("method", "unknown"),
        "accepted_requests": accepted,
        "total_requests": n_requests,
        "acceptance_rate": accepted / n_requests if n_requests else 0.0,
        "total_utility": total_utility,
        "avg_fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
        "avg_success_probability": sum(success_probs) / len(success_probs) if success_probs else None,
        "feasible": feas["feasible"],
        "edge_violations": feas["edge_violations"],
        "memory_violations": feas["memory_violations"],
        "mean_edge_utilization": sum(edge_util.values()) / len(edge_util) if edge_util else 0.0,
        "mean_memory_utilization": sum(mem_util.values()) / len(mem_util) if mem_util else 0.0,
    }
    if reference_utility:
        metrics["optimality_gap"] = (reference_utility - total_utility) / reference_utility
    return metrics


def compare_all(results: Dict[str, dict], bundles: List[dict], edge_capacities: Dict[Edge, int],
                 memory_capacities: Dict[str, int], all_request_ids: Optional[List[str]] = None,
                 reference: Optional[str] = "cp_sat_exact") -> Dict[str, dict]:
    """Runs compute_metrics for every {name: result} pair, using `reference`'s
    total_utility (if present in `results`) as the optimality-gap baseline."""
    reference_utility = None
    if reference and reference in results:
        reference_utility = results[reference]["total_utility"]

    return {
        name: compute_metrics(result, bundles, edge_capacities, memory_capacities,
                               all_request_ids=all_request_ids, reference_utility=reference_utility)
        for name, result in results.items()
    }


def print_comparison(metrics: Dict[str, dict]) -> None:
    """Pretty-print a comparison table sorted by total utility."""
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["total_utility"], reverse=True)
    header = f"{'method':28s} {'util':>9s} {'accept%':>8s} {'avgF':>6s} {'feas':>5s} {'gap%':>7s}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        gap = m.get("optimality_gap")
        gap_s = f"{gap * 100:6.1f}" if gap is not None else "   n/a"
        avg_f = f"{m['avg_fidelity']:.3f}" if m["avg_fidelity"] is not None else "  n/a"
        print(f"{name:28s} {m['total_utility']:9.3f} {m['acceptance_rate'] * 100:7.1f}% "
              f"{avg_f:>6s} {'yes' if m['feasible'] else 'NO':>5s} {gap_s:>7s}")
