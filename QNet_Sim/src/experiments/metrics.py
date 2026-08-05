import time
import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple


class ExperimentTracker:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def run_solver(self, solver_fn: Callable, bundles: List[dict],
                   edge_capacities: dict, memory_capacities: dict,
                   solver_name: str, instance_name: str = "",
                   **solver_kwargs) -> Dict[str, Any]:
        start = time.perf_counter()
        try:
            result = solver_fn(**solver_kwargs)
            elapsed = time.perf_counter() - start
        except Exception as e:
            elapsed = time.perf_counter() - start
            result = {"selected": [], "energy": float("inf"), "selections": {}}
            success = False
            error = str(e)
        else:
            success = True
            error = ""

        stats = self._compute_stats(result, bundles, edge_capacities,
                                     memory_capacities)
        record = {
            "solver": solver_name,
            "instance": instance_name,
            "wall_time_s": elapsed,
            "success": success,
            "error": error,
            **stats,
        }
        self.results.append(record)
        return record

    def _compute_stats(self, result: Dict, bundles: List[dict],
                       edge_capacities: dict,
                       memory_capacities: dict) -> Dict[str, Any]:
        selected = result.get("selected", [])
        energy = result.get("energy", float("inf"))

        n_requests = len(set(b["request_id"] for b in bundles))
        served_requests = len(selected)

        total_utility = 0.0
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        for rid, bid in selected:
            for b in bundles:
                if b["request_id"] == rid and b["bundle_id"] == bid:
                    total_utility += b["utility"]
                    for e, d in b["edge_demands"].items():
                        edge_load[tuple(sorted(e))] += d
                    for n, d in b["memory_demands"].items():
                        mem_load[n] += d
                    break

        violations = 0
        for edge, load in edge_load.items():
            cap = edge_capacities.get(edge, 0)
            if load > cap:
                violations += 1
        for node, load in mem_load.items():
            cap = memory_capacities.get(node, 0)
            if load > cap:
                violations += 1

        congestion_ratio = 0.0
        n_nonzero = 0
        for edge, load in edge_load.items():
            cap = edge_capacities.get(edge, 1)
            ratio = load / cap
            if ratio > 0:
                congestion_ratio += ratio
                n_nonzero += 1
        avg_congestion = congestion_ratio / max(n_nonzero, 1)

        return {
            "served": served_requests,
            "n_requests": n_requests,
            "served_ratio": served_requests / max(n_requests, 1),
            "total_utility": total_utility,
            "energy": energy,
            "violations": violations,
            "avg_congestion": avg_congestion,
            "n_selected_bundles": len(selected),
        }

    def summary(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        by_solver = defaultdict(list)
        for r in self.results:
            by_solver[r["solver"]].append(r)

        summary = {}
        for solver, records in by_solver.items():
            n = len(records)
            served = [r["served_ratio"] for r in records]
            utilities = [r["total_utility"] for r in records]
            times = [r["wall_time_s"] for r in records]
            violations = [r["violations"] for r in records]
            congestion = [r["avg_congestion"] for r in records]
            summary[solver] = {
                "n_runs": n,
                "mean_served_ratio": sum(served) / n,
                "mean_utility": sum(utilities) / n,
                "mean_time_s": sum(times) / n,
                "total_time_s": sum(times),
                "mean_violations": sum(violations) / n,
                "mean_congestion": sum(congestion) / n,
            }
        return summary

    def table(self) -> str:
        s = self.summary()
        if not s:
            return "(no results)"
        header = f"{'Solver':<25} {'Served':>8} {'Utility':>10} {'Time(s)':>10} {'Viol':>5} {'Cong':>8}"
        sep = "-" * len(header)
        lines = [header, sep]
        for solver, stats in sorted(s.items()):
            lines.append(
                f"{solver:<25} {stats['mean_served_ratio']:>8.3f} "
                f"{stats['mean_utility']:>10.2f} {stats['mean_time_s']:>10.4f} "
                f"{stats['mean_violations']:>5.1f} {stats['mean_congestion']:>8.3f}"
            )
        return "\n".join(lines)

    def to_csv(self, path: str):
        import csv
        if not self.results:
            return
        keys = list(self.results[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.results)
