"""Adaptive QUBO candidate budget (Wave 3, item 5).

The adaptive candidate-reduction stage keeps the top ``k`` bundles per request
before annealing.  Choosing ``k`` by hand is brittle: too small and the
optimizer can be starved of feasible alternatives on congested / strict-SLA
instances; too large and the QUBO is needlessly large and slow.  The review of
PR #8 already observed empirically that ``k = 4, 8, 16, 32`` often converge to
the same selected set while the QUBO size changes --- so the *correct* ``k``
depends on the network state.

This module makes the budget adaptive,

.. math::

    k = \\text{round}\\left(k_{base}\\,(1 + w_c\\,\\rho + w_d\\,d + w_f\\,f + w_m\\,\\mu)\\right)

where :math:`\\rho` is the (shadow) edge-congestion ratio, :math:`d` the
request density, :math:`f` the fidelity-strictness of the SLA distribution and
:math:`\\mu` the memory pressure --- all normalized to :math:`[0, 1]` --- and
``k`` is clamped to ``[k_min, k_max]``.

Direction of the policy: candidate diversity grows with request density and
SLA strictness (more genuinely different options worth keeping), whereas
congestion and memory pressure *contract* the budget --- under heavy load many
alternatives are infeasible or dominated anyway, and the QUBO annealer
performs best on a small, feasible-dominated pool (with fixed reads its
solution quality degrades as the QUBO grows).  ``run_adaptive_budget_study``
then compares fixed ``k`` against the adaptive policy on utility, QUBO size
and wall time against the exact ILP optimum, quantifying when candidate
reduction actually pays.
"""

import random
import time
from typing import Callable, Dict, List, Optional, Tuple


class AdaptiveBudgetPolicy:
    """Map a network state to a QUBO candidate budget ``k``."""

    def __init__(self, base: float = 4.0, k_min: int = 2, k_max: int = 32,
                 congestion_weight: float = -1.0,
                 density_weight: float = 0.75,
                 fidelity_weight: float = 1.0,
                 memory_weight: float = -0.75,
                 seed: int = 42):
        self.base = base
        self.k_min = k_min
        self.k_max = k_max
        self.congestion_weight = congestion_weight
        self.density_weight = density_weight
        self.fidelity_weight = fidelity_weight
        self.memory_weight = memory_weight
        self.seed = seed

    # ------------------------------------------------------------------
    def state_features(self, bundles: List[dict], edge_capacities: dict,
                       memory_capacities: dict) -> Dict[str, float]:
        """Normalized network-state features in ``[0, 1]`` (approximate)."""
        n_requests = len({b["request_id"] for b in bundles})
        n_edges = max(len(edge_capacities), 1)

        e_demand = {}
        m_demand = {}
        for b in bundles:
            for e, d in b["edge_demands"].items():
                e_demand[e] = e_demand.get(e, 0) + d
            for nd, d in b["memory_demands"].items():
                m_demand[nd] = m_demand.get(nd, 0) + d

        cong_ratios = []
        for e, cap in edge_capacities.items():
            load = e_demand.get(e, 0)
            cong_ratios.append(load / max(cap, 1))
        mem_ratios = []
        for nd, cap in memory_capacities.items():
            load = m_demand.get(nd, 0)
            mem_ratios.append(load / max(cap, 1))

        cong = min(1.0, sum(cong_ratios) / max(len(cong_ratios), 1))
        mem = min(1.0, sum(mem_ratios) / max(len(mem_ratios), 1))

        # request density relative to a reference of ~1 request per 2 edges
        density = min(1.0, n_requests / max(2 * n_edges, 1))

        # fidelity strictness: how demanding are the SLAs / decisions on average
        by_req = {}
        for b in bundles:
            by_req.setdefault(b["request_id"], []).append(b)
        if any("min_fidelity" in b for b in bundles):
            fvals = []
            for b in bundles:
                mf = b.get("min_fidelity", 0.5)
                fvals.append(max(0.0, min(1.0, (mf - 0.5) / 0.5)))
            fstrict = sum(fvals) / max(len(fvals), 1)
        else:
            spreads = []
            for bs in by_req.values():
                if not bs:
                    continue
                fids = [b.get("fidelity", 0.0) for b in bs]
                spreads.append(max(fids) - min(fids))
            fstrict = sum(spreads) / max(len(spreads), 1)

        return {"congestion": cong, "density": density,
                "fidelity_strictness": fstrict, "memory_pressure": mem}

    def select_k(self, bundles: List[dict], edge_capacities: dict,
                 memory_capacities: dict) -> int:
        """Compute the adaptive budget for the given candidate set."""
        s = self.state_features(bundles, edge_capacities, memory_capacities)
        k = self.base * (1.0
                         + self.congestion_weight * s["congestion"]
                         + self.density_weight * s["density"]
                         + self.fidelity_weight * s["fidelity_strictness"]
                         + self.memory_weight * s["memory_pressure"])
        return max(self.k_min, min(self.k_max, int(round(k))))


def _utility(selection: List[Tuple[str, str]], bundles: List[dict]) -> float:
    bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    return sum(bmap.get(k, {"utility": 0.0})["utility"] for k in selection)


def _solve_with_metropolis(bundles: List[dict], edge_capacities: dict,
                           memory_capacities: dict, k: int, seed: int) -> Dict:
    """Solve a k-reduced candidate set with the strong Metropolis annealer."""
    from optimization.adaptive_qubo import AdaptiveCandidateSelector
    from optimization.metropolis_annealer import MetropolisAnnealer
    from optimization.qubo_optimizer import QUBOOptimizer

    selector = AdaptiveCandidateSelector(edge_capacities, memory_capacities,
                                         seed=seed)
    reduced = selector.select(bundles, k=k) if k else bundles
    optimizer = QUBOOptimizer(reduced, edge_capacities, memory_capacities)
    n_vars = optimizer.to_bqm().num_variables

    annealer = MetropolisAnnealer(reduced, edge_capacities, memory_capacities,
                                  seed=seed)
    t0 = time.perf_counter()
    result = annealer.solve(
                            max_iterations=3000,
                            n_restarts=1, steps_per_temperature=10)
    elapsed = time.perf_counter() - t0

    util = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return {
        "k": k,
        "n_bundles_in": len(bundles),
        "n_bundles_in_qubo": len(reduced),
        "n_qubo_variables": n_vars,
        "utility": sum(util.get(s, 0.0) for s in result["selected"]),
        "served": len(result["selected"]),
        "wall_time_s": elapsed,
        "selected": result["selected"],
    }


def adaptive_budget_solve(bundles: List[dict], edge_capacities: dict,
                          memory_capacities: dict, num_reads: int = 20,
                          policy: Optional[AdaptiveBudgetPolicy] = None,
                          solver: str = "metropolis",
                          seed: int = 42) -> Dict:
    """Solve with the *adaptive* candidate budget (k chosen by policy)."""
    from experiments.optimality_benchmark import exact_ilp_solution

    if policy is None:
        policy = AdaptiveBudgetPolicy(seed=seed)
    k = policy.select_k(bundles, edge_capacities, memory_capacities)
    if solver == "metropolis":
        r = _solve_with_metropolis(bundles, edge_capacities, memory_capacities,
                                   k, seed=seed)
    else:
        from optimization.adaptive_qubo import adaptive_qubo_solve
        r = adaptive_qubo_solve(bundles, edge_capacities, memory_capacities,
                                k=k, num_reads=num_reads, seed=seed)
    ref_util = exact_ilp_solution(bundles, edge_capacities,
                                  memory_capacities)["u_star"]
    gap = max(0.0, ref_util - r["utility"]) if ref_util > 0 else 0.0
    return {**r, "gap_vs_full": gap,
            "relative_gap_vs_full": gap / max(ref_util, 1e-12)}


def run_adaptive_budget_study(topology_fn: Callable,
                              n_requests_list: Optional[List[int]] = None,
                              k_values: Optional[List[int]] = None,
                              num_reads: int = 20,
                              solver: str = "metropolis",
                              seed: int = 42) -> Dict:
    """Fixed-k vs adaptive-k comparison across several instances.

    Instances of increasing request count exercise progressively harder network
    states; the adaptive policy should grow ``k`` where the state warrants it
    while the fixed ``k`` stay constant.  Returns per-instance rows with
    ``method`` in ``k4..k32,adaptive`` carrying utility, QUBO size, wall time
    and the gap relative to the exact ILP optimum.
    """
    from experiments.instances import contention_sweep_instances
    if k_values is None:
        k_values = [4, 8, 16, 32]
    if n_requests_list is None:
        n_requests_list = [8, 12, 16, 20]

    insts = contention_sweep_instances(topology_fn, n_requests_list, seed=seed)

    policy = AdaptiveBudgetPolicy(seed=seed)
    rows = []
    summaries = []
    for n_req in n_requests_list:
        inst = insts[f"req{n_req}"]
        bundles, ec, mc = (inst["bundles"], inst["edge_capacities"],
                           inst["memory_capacities"])
        state = policy.state_features(bundles, ec, mc)
        k_adaptive = policy.select_k(bundles, ec, mc)

        inst_rows = []
        for k in k_values:
            if solver == "metropolis":
                r = _solve_with_metropolis(bundles, ec, mc, k, seed=seed)
            else:
                from optimization.adaptive_qubo import adaptive_qubo_solve
                r = adaptive_qubo_solve(bundles, ec, mc, k=k, num_reads=num_reads,
                                        seed=seed)
            inst_rows.append({
                "n_requests": n_req, "method": f"k{k}", "k": k,
                "utility": r["utility"], "served": r["served"],
                "n_qubo_variables": r["n_qubo_variables"],
                "wall_time_s": r["wall_time_s"],
            })
        # ground truth: exact optimum by scipy HiGHS ILP (tiny instances)
        from experiments.optimality_benchmark import exact_ilp_solution
        ref_util = exact_ilp_solution(bundles, ec, mc)["u_star"]
        for row in inst_rows:
            gap = max(0.0, ref_util - row["utility"])
            row["gap_vs_full"] = gap
            row["relative_gap_vs_full"] = gap / max(ref_util, 1e-12)

        r_adap = adaptive_budget_solve(bundles, ec, mc, num_reads=num_reads,
                                       policy=policy, solver=solver, seed=seed)
        adap_row = {
            "n_requests": n_req, "method": "adaptive", "k": k_adaptive,
            "utility": r_adap["utility"], "served": r_adap["served"],
            "n_qubo_variables": r_adap["n_qubo_variables"],
            "wall_time_s": r_adap["wall_time_s"],
        }
        gap = max(0.0, ref_util - r_adap["utility"])
        adap_row["gap_vs_full"] = gap
        adap_row["relative_gap_vs_full"] = gap / max(ref_util, 1e-12)
        inst_rows.append(adap_row)
        rows.extend(inst_rows)
        summaries.append({
            "n_requests": n_req, "n_bundles": len(bundles),
            "k_adaptive": k_adaptive, "state": state,
            "reference_utility": ref_util,
        })

    return {"n_requests_list": n_requests_list, "summaries": summaries,
            "rows": rows}
