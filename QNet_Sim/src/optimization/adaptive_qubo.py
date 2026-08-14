"""Adaptive candidate-space reduction for the QUBO (Extension 7).

Instead of building the QUBO over every generated bundle, the candidate set is
reduced *from the current network state* before the QUBO is compiled:

    1000 possible paths -> fidelity filter -> latency filter ->
    memory-feasibility check -> state-aware top-k -> QUBO

``run_topk_sweep`` sweeps k in {2, 4, 8, 16, 32} and reports runtime, solution
quality, QUBO size and the optimality gap against the full-candidate reference,
quantifying how aggressively the search space can be pruned without losing
solution quality (the "QiOpt4QNet scales through adaptive reduction" result).
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple


def _undirected(edge):
    return tuple(sorted(edge))


class AdaptiveCandidateSelector:
    """State-aware top-k candidate selection per request.

    The selection uses a *shadow routing* estimate of the network state: for
    each request its currently-best candidate (by utility density) is placed,
    edge and memory loads are computed, and every candidate is then scored by
    ``utility / (1 + w * resource_density)`` where ``resource_density`` counts
    how much of an already-loaded resource it would consume.  The top-k
    candidates per request survive to the QUBO.
    """

    def __init__(self, edge_capacities: dict, memory_capacities: dict,
                 fidelity_threshold: float = 0.0,
                 max_latency: float = float("inf"),
                 congestion_weight: float = 0.5, seed: int = 42):
        self.edge_capacities = {_undirected(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = dict(memory_capacities)
        self.fidelity_threshold = fidelity_threshold
        self.max_latency = max_latency
        self.congestion_weight = congestion_weight
        self.seed = seed

    def select(self, bundles: List[dict], k: int = 8) -> List[dict]:
        if k is None or k <= 0:
            return bundles
        by_req: Dict[str, List[dict]] = defaultdict(list)
        for b in bundles:
            by_req[b["request_id"]].append(b)

        # --- shadow routing: one preferred bundle per request ---
        preferred: Dict[str, dict] = {}
        for rid, bs in by_req.items():
            pool = [b for b in bs if self._passes(b)]
            if not pool:
                continue
            preferred[rid] = max(pool, key=lambda b: self._density(b, {}))

        edge_load: Dict[tuple, int] = defaultdict(int)
        mem_load: Dict[str, int] = defaultdict(int)
        for rid, b in preferred.items():
            for e, d in b["edge_demands"].items():
                edge_load[_undirected(e)] += d
            for n, d in b["memory_demands"].items():
                mem_load[n] += d

        # --- score each candidate against the shadow state, keep top-k ---
        out: List[dict] = []
        for rid, bs in by_req.items():
            scored = []
            for b in bs:
                if not self._passes(b):
                    continue
                if not self._individually_feasible(b):
                    continue
                score = self._density(b, edge_load, mem_load)
                scored.append((score, b))
            scored.sort(key=lambda x: x[0], reverse=True)
            out.extend(b[1] for b in scored[:k])
        return out

    def _passes(self, b: dict) -> bool:
        if b.get("fidelity", 0.0) < self.fidelity_threshold:
            return False
        if b.get("latency", 0.0) > self.max_latency:
            return False
        return True

    def _individually_feasible(self, b: dict) -> bool:
        """A bundle that exceeds a node capacity on its own can never be used."""
        for n, d in b.get("memory_demands", {}).items():
            if d > self.memory_capacities.get(n, 0):
                return False
        for e, d in b.get("edge_demands", {}).items():
            if d > self.edge_capacities.get(_undirected(e), 0):
                return False
        return True

    def _density(self, b: dict, edge_load: Dict[tuple, int],
                 mem_load: Optional[Dict[str, int]] = None) -> float:
        utility = b.get("utility", 0.0)
        res = 0.0
        for e, d in b.get("edge_demands", {}).items():
            e = _undirected(e)
            cap = self.edge_capacities.get(e, 1) or 1
            res += (edge_load.get(e, 0) + d) / cap
        if mem_load is not None:
            for n, d in b.get("memory_demands", {}).items():
                cap = self.memory_capacities.get(n, 1) or 1
                res += (mem_load.get(n, 0) + d) / cap
        else:
            for n, d in b.get("memory_demands", {}).items():
                cap = self.memory_capacities.get(n, 1) or 1
                res += d / cap
        return utility / (1.0 + self.congestion_weight * res)


def _utility(bundles: List[dict], selected: List[Tuple[str, str]]) -> float:
    util = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return sum(util.get(k, 0.0) for k in selected)


def adaptive_qubo_solve(bundles: List[dict], edge_capacities: dict,
                        memory_capacities: dict, k: int = 8,
                        num_reads: int = 30, seed: int = 42) -> Dict:
    """Reduced-candidate QUBO solve (with feasibility repair)."""
    import time
    from optimization.qubo_optimizer import QUBOOptimizer
    from optimization.openjij_solver import solve_sa

    selector = AdaptiveCandidateSelector(edge_capacities, memory_capacities,
                                         seed=seed)
    reduced = selector.select(bundles, k=k)

    t0 = time.perf_counter()
    optimizer = QUBOOptimizer(reduced, edge_capacities, memory_capacities)
    bqm = optimizer.to_bqm(
                           congestion_penalty=0.0, memory_congestion_penalty=0.0)
    response = solve_sa(bqm, num_reads=num_reads, seed=seed)
    selected = optimizer.decode_sample(response.first.sample, repair=True)
    elapsed = time.perf_counter() - t0

    return {
        "k": k,
        "n_bundles_in": len(bundles),
        "n_bundles_in_qubo": len(reduced),
        "n_qubo_variables": bqm.num_variables,
        "utility": _utility(bundles, selected),
        "served": len(set(k_ for k_ in selected)),
        "wall_time_s": elapsed,
        "selected": selected,
    }


def reference_solution(bundles: List[dict], edge_capacities: dict,
                       memory_capacities: dict, num_reads: int = 60,
                       seed: int = 42) -> Dict:
    """Full-candidate-set QUBO reference (the 'no reduction' baseline)."""
    return adaptive_qubo_solve(bundles, edge_capacities, memory_capacities,
                               k=None, num_reads=num_reads, seed=seed)


def run_topk_sweep(topology_fn: Callable, n_requests: int = 16,
                   k_values: Optional[List[int]] = None,
                   num_reads: int = 30, seed: int = 42) -> Dict:
    """Sweep k and report runtime / quality / QUBO size / optimality gap."""
    from experiments.instances import contention_sweep_instances
    if k_values is None:
        k_values = [2, 4, 8, 16, 32]
    inst = contention_sweep_instances(topology_fn, [n_requests], seed=seed)[f"req{n_requests}"]
    bundles, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

    ref = reference_solution(bundles, ec, mc, num_reads=num_reads, seed=seed)
    ref_utility = ref["utility"]

    rows = []
    for k in k_values:
        r = adaptive_qubo_solve(bundles, ec, mc, k=k, num_reads=num_reads, seed=seed)
        r["optimality_gap"] = max(0.0, ref_utility - r["utility"]) if ref_utility > 0 else 0.0
        r["relative_gap"] = (r["optimality_gap"] / max(ref_utility, 1e-12)
                             if ref_utility > 0 else 0.0)
        rows.append(r)

    return {
        "n_requests": n_requests,
        "n_bundles": len(bundles),
        "reference": ref,
        "rows": rows,
    }
