"""Hybrid classical + quantum-inspired optimization pipeline (Extension 6).

The pipeline separates the classical pre-processing / post-processing from
the QUBO solve so that the QUBO only ever sees a small, well-chosen candidate
set and its output is always returned capacity-feasible:

    Stage 1  candidate reduction   -- classical filters (fidelity, latency,
                                      memory footprint) + dominance pruning;
    Stage 2  QUBO solve            -- PyQUBO + OpenJij simulated annealing on
                                      the *reduced* candidate set;
    Stage 3  feasibility repair    -- deterministic greedy repair drops
                                      conflicting bundles until feasible;
    Stage 4  local refinement      -- Metropolis warm-started from the repaired
                                      selection improves the objective without
                                      leaving feasibility.

``run_hybrid_comparison`` benchmarks the full pipeline against its ablations
(QUBO-only, repair-only, refine-only) so each stage's contribution is
measurable.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple


def _undirected(edge):
    return tuple(sorted(edge))


class CandidateReducer:
    """Stage 1: shrink the candidate set before the QUBO is built.

    Filters are hard gates (a bundle that violates one is removed); dominance
    pruning then removes bundles that are Pareto-dominated in
    (fidelity, bell-pair cost, latency) by a cheaper-or-better alternative.
    """

    def __init__(self, fidelity_threshold: float = 0.0,
                 max_latency: float = float("inf"),
                 max_bell_pair_cost: float = float("inf"),
                 max_memory_demand: float = float("inf"),
                 use_dominance_pruning: bool = True):
        self.fidelity_threshold = fidelity_threshold
        self.max_latency = max_latency
        self.max_bell_pair_cost = max_bell_pair_cost
        self.max_memory_demand = max_memory_demand
        self.use_dominance_pruning = use_dominance_pruning

    def reduce(self, bundles: List[dict], keep_per_request: Optional[int] = None,
               score_key: str = "utility") -> List[dict]:
        filtered = []
        for b in bundles:
            if b.get("fidelity", 0.0) < self.fidelity_threshold:
                continue
            if b.get("latency", 0.0) > self.max_latency:
                continue
            if b.get("bell_pair_cost", float("inf")) > self.max_bell_pair_cost:
                continue
            mem_demand = sum(b.get("memory_demands", {}).values())
            if mem_demand > self.max_memory_demand:
                continue
            filtered.append(b)
        if self.use_dominance_pruning:
            filtered = _dominance_prune(filtered)
        if keep_per_request is not None and keep_per_request > 0:
            by_req: Dict[str, List[dict]] = defaultdict(list)
            for b in filtered:
                by_req[b["request_id"]].append(b)
            out = []
            for rid, bs in by_req.items():
                bs.sort(key=lambda b: b.get(score_key, 0.0), reverse=True)
                out.extend(bs[:keep_per_request])
            return out
        return filtered


def _dominance_prune(bundles: List[dict]) -> List[dict]:
    """Remove bundles dominated on (fidelity, cost, latency)."""
    pruned = []
    for i, bc in enumerate(bundles):
        dominated = False
        for j, bo in enumerate(bundles):
            if i == j:
                continue
            no_worse = (bo.get("fidelity", 0.0) >= bc.get("fidelity", 0.0)
                        and bo.get("bell_pair_cost", float("inf")) <= bc.get("bell_pair_cost", float("inf"))
                        and bo.get("latency", float("inf")) <= bc.get("latency", float("inf")))
            strict = (bo.get("fidelity", 0.0) > bc.get("fidelity", 0.0)
                      or bo.get("bell_pair_cost", float("inf")) < bc.get("bell_pair_cost", float("inf"))
                      or bo.get("latency", float("inf")) < bc.get("latency", float("inf")))
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            pruned.append(bc)
    return pruned


class HybridPipeline:
    """Four-stage hybrid optimizer (Extension 6)."""

    def __init__(self, bundles: List[dict], edge_capacities: dict,
                 memory_capacities: dict, reducer: Optional[CandidateReducer] = None,
                 seed: int = 42):
        self.all_bundles = bundles
        self.edge_capacities = edge_capacities
        self.memory_capacities = memory_capacities
        self.reducer = reducer or CandidateReducer()
        self.seed = seed
        self.stats: Dict = {}

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def stage1_reduce(self, keep_per_request: Optional[int] = None) -> List[dict]:
        reduced = self.reducer.reduce(self.all_bundles, keep_per_request=keep_per_request)
        self.stats["n_candidates_in"] = len(self.all_bundles)
        self.stats["n_candidates_out"] = len(reduced)
        return reduced

    def stage2_qubo(self, bundles: List[dict], num_reads: int = 50,
                    penalty: float = 100.0, edge_penalty: float = 10.0,
                    memory_penalty: float = 10.0) -> List[Tuple[str, str]]:
        """Build a PyQUBO model on the reduced set and solve with OpenJij SA."""
        from optimization.qubo_optimizer import QUBOOptimizer
        from optimization.openjij_solver import solve_sa
        optimizer = QUBOOptimizer(bundles, self.edge_capacities, self.memory_capacities)
        bqm = optimizer.to_bqm(penalty=penalty, edge_penalty=edge_penalty,
                               memory_penalty=memory_penalty,
                               congestion_penalty=0.0, memory_congestion_penalty=0.0)
        response = solve_sa(bqm, num_reads=num_reads, seed=self.seed)
        return optimizer.decode_sample(response.first.sample, repair=False)

    def stage3_repair(self, bundles: List[dict],
                      selected: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Deterministic greedy feasibility repair (never infeasible)."""
        from optimization.qubo_optimizer import QUBOOptimizer
        optimizer = QUBOOptimizer(bundles, self.edge_capacities, self.memory_capacities)
        return optimizer.repair_selection(selected)

    def stage4_refine(self, bundles: List[dict],
                      selected: List[Tuple[str, str]],
                      max_iterations: int = 3000) -> List[Tuple[str, str]]:
        """Metropolis local refinement warm-started from a feasible selection."""
        from optimization.metropolis_annealer import MetropolisAnnealer
        selections = {rid: None for rid in {b["request_id"] for b in bundles}}
        for rid, bid in selected:
            selections[rid] = bid
        annealer = MetropolisAnnealer(bundles, self.edge_capacities,
                                      self.memory_capacities, seed=self.seed)
        result = annealer.solve(penalty=100.0, edge_penalty=10.0,
                                memory_penalty=10.0, max_iterations=max_iterations,
                                n_restarts=1, initial_selections=selections)
        return result["selected"]

    # ------------------------------------------------------------------
    # orchestration
    # ------------------------------------------------------------------
    def solve(self, keep_per_request: Optional[int] = None, num_reads: int = 50,
              refine_iterations: int = 3000) -> Dict:
        import time
        from experiments.metrics import ExperimentTracker

        tracker = ExperimentTracker()

        t0 = time.perf_counter()
        reduced = self.stage1_reduce(keep_per_request=keep_per_request)
        self.stats["stage1_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        raw = self.stage2_qubo(reduced, num_reads=num_reads)
        self.stats["stage2_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        repaired = self.stage3_repair(reduced, raw)
        self.stats["stage3_time_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        refined = self.stage4_refine(reduced, repaired,
                                     max_iterations=refine_iterations)
        self.stats["stage4_time_s"] = time.perf_counter() - t0

        tracker.run_solver(lambda: {"selected": refined}, reduced,
                           self.edge_capacities, self.memory_capacities,
                           solver_name="hybrid_pipeline", instance_name="")
        record = tracker.results[0]
        self.stats["served"] = record["served"]
        self.stats["total_utility"] = record["total_utility"]
        self.stats["violations"] = record["violations"]
        return {"selected": refined, "stats": self.stats}


def _utility_of(bundles: List[dict], selected: List[Tuple[str, str]]) -> float:
    util = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return sum(util.get(k, 0.0) for k in selected)


def run_hybrid_comparison(topology_fn: Callable, n_requests: int = 12,
                          seed: int = 42) -> Dict:
    """Benchmark the full pipeline against its stage ablations."""
    from experiments.instances import contention_sweep_instances
    inst = contention_sweep_instances(topology_fn, [n_requests], seed=seed)["n%d" % n_requests]
    bundles, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

    pipeline = HybridPipeline(bundles, ec, mc, seed=seed)
    full = pipeline.solve(keep_per_request=8, num_reads=30, refine_iterations=1500)

    reduced = pipeline.stage1_reduce(keep_per_request=8)
    raw = pipeline.stage2_qubo(reduced, num_reads=30)
    repaired = pipeline.stage3_repair(reduced, raw)

    return {
        "n_bundles_in": len(bundles),
        "n_bundles_reduced": len(reduced),
        "full_pipeline": {
            "utility": _utility_of(bundles, full["selected"]),
            "served": full["stats"]["served"],
            "violations": full["stats"]["violations"],
            "stats": full["stats"],
        },
        "qubo_only": {
            "utility": _utility_of(bundles, raw),
            "served": len(set(k[0] for k in raw)),
        },
        "qubo_plus_repair": {
            "utility": _utility_of(bundles, repaired),
            "served": len(set(k[0] for k in repaired)),
        },
    }
