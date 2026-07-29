"""
Classical (non-quantum) route/bundle allocation baselines for QiOpt4QNet.

Every allocator here shares the *exact* interface already used by the
quantum/annealing optimizers in `optimization/` (QUBOOptimizer,
MetropolisAnnealer, TensorNetworkOptimizer):

    allocator = SomeAllocator(bundles, edge_capacities, memory_capacities)
    result = allocator.solve()
    result["selected"]        -> list[(request_id, bundle_id)]
    result["total_utility"]   -> float

`bundles` is a list of dicts. At minimum each needs the keys produced by
`Bundle.to_optimizer_dict()`:
    bundle_id, request_id, path, edge_demands, memory_demands, utility
Baselines that rank by fidelity / success probability / bell-pair cost
will use those fields directly if you pass the richer `Bundle.to_dict()`
output instead, and fall back to a resource-demand approximation if they
are absent. This lets classical and quantum solvers be swapped into the
same evaluation harness (see feasibility.py) without any glue code.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from ortools.sat.python import cp_model
    CPSAT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when ortools isn't installed
    CPSAT_AVAILABLE = False


Edge = Tuple[str, str]
BundleKey = Tuple[str, str]  # (request_id, bundle_id)


def _undirected(edge: Tuple[str, str]) -> Edge:
    return tuple(sorted(edge))


class BaseAllocator(ABC):
    """Common interface shared with the QUBO / annealing optimizers."""

    name = "base"
    _required = {"bundle_id", "request_id", "path", "edge_demands", "memory_demands", "utility"}

    def __init__(self, bundles: List[dict], edge_capacities: Dict[Edge, int],
                 memory_capacities: Dict[str, int], seed: Optional[int] = None):
        self.bundles = bundles
        self.edge_capacities = {_undirected(e): c for e, c in edge_capacities.items()}
        self.memory_capacities = dict(memory_capacities)
        self.rng = random.Random(seed)
        self._validate()
        self._group_by_request()

    def _validate(self):
        for b in self.bundles:
            missing = self._required - b.keys()
            if missing:
                raise ValueError(f"Bundle {b.get('bundle_id', '?')} missing: {missing}")

    def _group_by_request(self):
        self.requests: List[str] = []
        self.bundles_per_request: Dict[str, List[dict]] = {}
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in self.bundles_per_request:
                self.bundles_per_request[rid] = []
                self.requests.append(rid)
            self.bundles_per_request[rid].append(b)

    # ---- helpers shared by every scoring strategy ----
    @staticmethod
    def _edges(b: dict) -> Dict[Edge, int]:
        return {_undirected(e): d for e, d in b["edge_demands"].items()}

    @staticmethod
    def _resource_cost(b: dict) -> float:
        """Total resource footprint of a bundle (edge + memory demand)."""
        return sum(b["edge_demands"].values()) + sum(b["memory_demands"].values())

    def _fits(self, b: dict, edge_load: Dict[Edge, int], mem_load: Dict[str, int]) -> bool:
        for edge, d in self._edges(b).items():
            if edge_load[edge] + d > self.edge_capacities.get(edge, 0):
                return False
        for node, d in b["memory_demands"].items():
            if mem_load[node] + d > self.memory_capacities.get(node, 0):
                return False
        return True

    def _commit(self, b: dict, edge_load: Dict[Edge, int], mem_load: Dict[str, int]):
        for edge, d in self._edges(b).items():
            edge_load[edge] += d
        for node, d in b["memory_demands"].items():
            mem_load[node] += d

    def _first_fit_by(self, key) -> dict:
        edge_load: Dict[Edge, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)
        selections: Dict[str, Optional[str]] = {}
        for rid in self.requests:
            pool = sorted(self.bundles_per_request[rid], key=key)
            chosen = None
            for b in pool:
                if self._fits(b, edge_load, mem_load):
                    chosen = b
                    self._commit(b, edge_load, mem_load)
                    break
            selections[rid] = chosen["bundle_id"] if chosen else None
        return self._finalize(selections)

    def _finalize(self, selections: Dict[str, Optional[str]]) -> dict:
        by_key = {(b["request_id"], b["bundle_id"]): b for b in self.bundles}
        selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        total_utility = sum(by_key[(rid, bid)]["utility"] for rid, bid in selected)
        return {
            "selected": selected,
            "selections": selections,
            "total_utility": total_utility,
            "method": self.name,
        }

    @abstractmethod
    def solve(self) -> dict:
        ...


# 1. Random feasible allocation
class RandomFeasibleAllocator(BaseAllocator):

    name = "random_feasible"

    def solve(self) -> dict:
        order = list(self.requests)
        self.rng.shuffle(order)
        edge_load: Dict[Edge, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)
        selections: Dict[str, Optional[str]] = {}
        for rid in order:
            pool = list(self.bundles_per_request[rid])
            self.rng.shuffle(pool)
            chosen = None
            for b in pool:
                if self._fits(b, edge_load, mem_load):
                    chosen = b
                    self._commit(b, edge_load, mem_load)
                    break
            selections[rid] = chosen["bundle_id"] if chosen else None
        return self._finalize(selections)


# 2. Simple per-request heuristics ("X-first")
class ShortestFeasiblePathAllocator(BaseAllocator):
    """Prefers fewest hops, then higher utility as a tiebreak."""

    name = "shortest_feasible_path"

    def solve(self) -> dict:
        return self._first_fit_by(key=lambda b: (len(b.get("path", [])), -b["utility"]))


class HighestFidelityFirstAllocator(BaseAllocator):
    name = "highest_fidelity_first"

    def solve(self) -> dict:
        return self._first_fit_by(key=lambda b: -b.get("fidelity", 0.0))


class HighestSuccessFirstAllocator(BaseAllocator):
    name = "highest_success_first"

    def solve(self) -> dict:
        return self._first_fit_by(key=lambda b: -b.get("success_probability", 0.0))


class LowestResourceCostFirstAllocator(BaseAllocator):
    name = "lowest_resource_cost_first"

    def solve(self) -> dict:
        return self._first_fit_by(key=lambda b: b.get("bell_pair_cost", self._resource_cost(b)))


# 3. Global greedy strategies
class UtilityPerResourceGreedyAllocator(BaseAllocator):

    name = "utility_per_resource_greedy"

    def solve(self) -> dict:
        scored = sorted(self.bundles, key=lambda b: b["utility"] / (self._resource_cost(b) + 1e-9), reverse=True)
        edge_load: Dict[Edge, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)
        selections: Dict[str, Optional[str]] = {rid: None for rid in self.requests}
        assigned = set()
        for b in scored:
            rid = b["request_id"]
            if rid in assigned:
                continue
            if self._fits(b, edge_load, mem_load):
                selections[rid] = b["bundle_id"]
                self._commit(b, edge_load, mem_load)
                assigned.add(rid)
        return self._finalize(selections)


class CongestionAwareGreedyAllocator(BaseAllocator):

    name = "congestion_aware_greedy"

    def solve(self) -> dict:
        edge_load: Dict[Edge, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)
        selections: Dict[str, Optional[str]] = {rid: None for rid in self.requests}
        remaining = set(self.requests)

        while remaining:
            best_rid, best_bundle, best_score = None, None, float("-inf")
            for b in self.bundles:
                rid = b["request_id"]
                if rid not in remaining or not self._fits(b, edge_load, mem_load):
                    continue
                congestion = 1.0
                for edge, d in self._edges(b).items():
                    cap = self.edge_capacities.get(edge, 1) or 1
                    congestion += (edge_load[edge] + d) / cap
                for node, d in b["memory_demands"].items():
                    cap = self.memory_capacities.get(node, 1) or 1
                    congestion += (mem_load[node] + d) / cap
                score = b["utility"] / congestion
                if score > best_score:
                    best_rid, best_bundle, best_score = rid, b, score

            if best_bundle is None:
                break  # nothing left is feasible

            selections[best_rid] = best_bundle["bundle_id"]
            self._commit(best_bundle, edge_load, mem_load)
            remaining.discard(best_rid)

        return self._finalize(selections)


# 4. Greedy + local search
class GreedyLocalSearchAllocator(BaseAllocator):

    name = "greedy_local_search"

    def __init__(self, bundles, edge_capacities, memory_capacities,
                 max_passes: int = 10, seed: Optional[int] = None):
        super().__init__(bundles, edge_capacities, memory_capacities, seed=seed)
        self.max_passes = max_passes

    def solve(self) -> dict:
        seed_result = UtilityPerResourceGreedyAllocator(
            self.bundles, self.edge_capacities, self.memory_capacities
        ).solve()

        edge_load: Dict[Edge, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)
        current: Dict[str, Optional[dict]] = {}
        for rid, bid in seed_result["selections"].items():
            b = next((x for x in self.bundles_per_request[rid] if x["bundle_id"] == bid), None)
            current[rid] = b
            if b:
                self._commit(b, edge_load, mem_load)

        for _ in range(self.max_passes):
            improved = False
            order = list(self.requests)
            self.rng.shuffle(order)
            for rid in order:
                prev = current[rid]
                if prev:
                    for edge, d in self._edges(prev).items():
                        edge_load[edge] -= d
                    for node, d in prev["memory_demands"].items():
                        mem_load[node] -= d

                best = prev
                best_utility = prev["utility"] if prev else 0.0
                for cand in self.bundles_per_request.get(rid, []):
                    if cand is prev or cand["utility"] <= best_utility:
                        continue
                    if self._fits(cand, edge_load, mem_load):
                        best, best_utility = cand, cand["utility"]

                if best is not prev:
                    improved = True
                current[rid] = best
                if best:
                    self._commit(best, edge_load, mem_load)

            if not improved:
                break

        selections = {rid: (b["bundle_id"] if b else None) for rid, b in current.items()}
        return self._finalize(selections)


# 5. Exact reference solver (oracle)
class CPSATAllocator(BaseAllocator):
    """Exact MILP solve via OR-Tools CP-SAT: maximize total utility subject
    to one-bundle-per-request, edge capacity, and memory capacity
    constraints. This is the reference/oracle solution the doc calls for,
    and also the supervision source for the ML ranking model's training
    data (see dataset_generator.py)."""

    name = "cp_sat_exact"

    def __init__(self, bundles, edge_capacities, memory_capacities,
                 time_limit_s: float = 30.0, utility_scale: int = 1000, seed: Optional[int] = None):
        if not CPSAT_AVAILABLE:
            raise ImportError(
                "CPSATAllocator requires OR-Tools. Install with: "
                "pip install ortools --break-system-packages"
            )
        super().__init__(bundles, edge_capacities, memory_capacities, seed=seed)
        self.time_limit_s = time_limit_s
        self.utility_scale = utility_scale

    def solve(self) -> dict:
        model = cp_model.CpModel()
        x = {
            (b["request_id"], b["bundle_id"]): model.NewBoolVar(f"x_{i}")
            for i, b in enumerate(self.bundles)
        }

        for rid, bl in self.bundles_per_request.items():
            model.Add(sum(x[(rid, b["bundle_id"])] for b in bl) <= 1)

        edge_uses: Dict[Edge, list] = defaultdict(list)
        mem_uses: Dict[str, list] = defaultdict(list)
        for b in self.bundles:
            key = (b["request_id"], b["bundle_id"])
            for edge, d in self._edges(b).items():
                edge_uses[edge].append((key, d))
            for node, d in b["memory_demands"].items():
                mem_uses[node].append((key, d))

        for edge, uses in edge_uses.items():
            model.Add(sum(d * x[k] for k, d in uses) <= self.edge_capacities.get(edge, 0))
        for node, uses in mem_uses.items():
            model.Add(sum(d * x[k] for k, d in uses) <= self.memory_capacities.get(node, 0))

        model.Maximize(sum(
            int(round(b["utility"] * self.utility_scale)) * x[(b["request_id"], b["bundle_id"])]
            for b in self.bundles
        ))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_s
        status = solver.Solve(model)

        selections: Dict[str, Optional[str]] = {rid: None for rid in self.requests}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for b in self.bundles:
                key = (b["request_id"], b["bundle_id"])
                if solver.Value(x[key]) == 1:
                    selections[b["request_id"]] = b["bundle_id"]

        result = self._finalize(selections)
        result["status"] = solver.StatusName(status)
        result["is_optimal"] = status == cp_model.OPTIMAL
        return result


# Registry + convenience runner
ALL_BASELINES = {
    "random_feasible": RandomFeasibleAllocator,
    "shortest_feasible_path": ShortestFeasiblePathAllocator,
    "highest_fidelity_first": HighestFidelityFirstAllocator,
    "highest_success_first": HighestSuccessFirstAllocator,
    "lowest_resource_cost_first": LowestResourceCostFirstAllocator,
    "utility_per_resource_greedy": UtilityPerResourceGreedyAllocator,
    "congestion_aware_greedy": CongestionAwareGreedyAllocator,
    "greedy_local_search": GreedyLocalSearchAllocator,
}
if CPSAT_AVAILABLE:
    ALL_BASELINES["cp_sat_exact"] = CPSATAllocator


def run_all_baselines(bundles: List[dict], edge_capacities: Dict[Edge, int],
                       memory_capacities: Dict[str, int], seed: Optional[int] = None,
                       include: Optional[List[str]] = None) -> Dict[str, dict]:
    """Runs every registered baseline (optionally a subset via `include`) on
    the same instance and returns {name: result}."""
    names = include or list(ALL_BASELINES.keys())
    results = {}
    for name in names:
        allocator = ALL_BASELINES[name](bundles, edge_capacities, memory_capacities, seed=seed)
        results[name] = allocator.solve()
    return results
