"""Adaptive recourse: local repair versus full reoptimization (Wave 3, item 3).

A plan ``{(rid, bid)}`` is executed by the stochastic engine; in a given
realization some requests fail (purification / swap / memory).  An operator
has two responses:

* **full reoptimization** -- re-run the solver over *all* requests from
  scratch (``FullReoptimizer``);
* **local repair** -- commit the requests that survived, free the resources of
  the failed ones, and solve a *small* QUBO over only the failed requests'
  candidate bundles against the residual capacities (``LocalRepair``).

The experiment ``run_recourse_comparison`` evaluates both on the same
stochastic realizations and reports the wall-clock ratio ``T_full / T_local``,
the recovered utility, and the fraction of failed requests re-routed
successfully.  The claim this supports: repairing only the affected portion
recovers most of the lost utility in a small fraction of the reoptimization
time, which is exactly the online-recourse property the hybrid pipeline needs.
"""

import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from experiments.instances import generate_chain_topology


def _loads(selection: List[Tuple[str, str]],
           bundles: List[dict]) -> Tuple[Dict, Dict]:
    """Aggregate edge/memory loads of a committed selection."""
    bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    e_load = defaultdict(float)
    m_load = defaultdict(float)
    for rid, bid in selection:
        b = bmap.get((rid, bid))
        if b is None:
            continue
        for e, d in b["edge_demands"].items():
            e_load[tuple(sorted(e))] += d
        for n, d in b["memory_demands"].items():
            m_load[n] += d
    return e_load, m_load


class LocalRepair:
    """Re-solve only the failed requests against residual capacity.

    The survived requests' resource consumption is fixed; the failed requests'
    resource usage is released, and a small QUBO over the failed requests'
    candidate bundles is solved with the *residual* edge/memory capacities.
    """

    def __init__(self, bundles: List[dict], edge_capacities: dict,
                 memory_capacities: dict, num_reads: int = 20, seed: int = 42):
        self.bundles = bundles
        self.edge_capacities = edge_capacities
        self.memory_capacities = memory_capacities
        self.num_reads = num_reads
        self.seed = seed
        self.by_request: Dict[str, List[dict]] = defaultdict(list)
        for b in bundles:
            self.by_request[b["request_id"]].append(b)

    def repair(self, committed: List[Tuple[str, str]],
               failed: List[str]) -> Tuple[List[Tuple[str, str]], float]:
        """Return (repaired selection, wall time) for the failed requests.

        ``committed`` is the surviving part of the plan; ``failed`` is the
        list of request ids that must be re-routed.  Requests that cannot be
        re-routed feasibly stay failed.
        """
        import time
        from optimization.qubo_optimizer import QUBOOptimizer
        from optimization.openjij_solver import solve_sa

        e_load, m_load = _loads(committed, self.bundles)
        residual_e = {e: int(self.edge_capacities.get(e, 0) - e_load.get(e, 0))
                      for e in self.edge_capacities}
        residual_m = {n: int(self.memory_capacities.get(n, 0) - m_load.get(n, 0))
                      for n in self.memory_capacities}

        reduced = []
        for rid in failed:
            reduced.extend(self.by_request.get(rid, []))
        if not reduced:
            return list(committed), 0.0

        t0 = time.perf_counter()
        opt = QUBOOptimizer(reduced, residual_e, residual_m)
        bqm = opt.to_bqm(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                         congestion_penalty=0.0, memory_congestion_penalty=0.0)
        resp = solve_sa(bqm, num_reads=self.num_reads, seed=self.seed)
        repaired = opt.decode_sample(resp.first.sample, repair=True)
        elapsed = time.perf_counter() - t0
        return list(committed) + repaired, elapsed


class FullReoptimizer:
    """Re-run the full solver over the entire instance."""

    def __init__(self, bundles: List[dict], edge_capacities: dict,
                 memory_capacities: dict, seed: int = 42):
        self.bundles = bundles
        self.edge_capacities = edge_capacities
        self.memory_capacities = memory_capacities
        self.seed = seed

    def reoptimize(self) -> Tuple[List[Tuple[str, str]], float]:
        from optimization.metropolis_annealer import MetropolisAnnealer
        opt = MetropolisAnnealer(self.bundles, self.edge_capacities,
                                 self.memory_capacities, seed=self.seed)
        t0 = time.perf_counter()
        r = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        elapsed = time.perf_counter() - t0
        return r.get("selected", []), elapsed


def _utility(selection: List[Tuple[str, str]], bundles: List[dict]) -> float:
    bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    return sum(bmap.get(k, {"utility": 0.0})["utility"] for k in selection)


def run_recourse_comparison(topology: dict, n_requests: int = 8,
                            n_realizations: int = 40, seed: int = 42,
                            tau_mem: float = 50.0,
                            swap_success: float = 0.95) -> Dict:
    """Compare local repair vs full reoptimization on identical failures.

    Returns aggregate rows plus a per-realization breakdown.  The local
    repair re-solves only the failed requests against residual capacity; the
    full reoptimizer re-runs the annealer over everything.
    """
    from experiments.instances import generate_benchmark_instance
    from optimization.metropolis_annealer import MetropolisAnnealer
    from simulation.discrete_event_engine import StochasticEventSimulator

    rng = random.Random(seed)
    nodes = topology["nodes"]
    pairs = []
    for _ in range(n_requests):
        src, dst = rng.sample(nodes, 2)
        pairs.append((src, dst, rng.uniform(10.0, 100.0), rng.uniform(0.5, 0.75)))

    bundles, ec, mc = generate_benchmark_instance(topology, pairs, rng)

    opt = MetropolisAnnealer(bundles, ec, mc, seed=seed)
    t0 = time.perf_counter()
    p0 = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                   max_iterations=3000, n_restarts=1, steps_per_temperature=10)
    plan0 = p0.get("selected", [])
    t_plan = time.perf_counter() - t0
    util0 = _utility(plan0, bundles)

    sim = StochasticEventSimulator(topology, tau_mem=tau_mem,
                                   swap_success=swap_success, seed=seed)
    agg = sim.simulate_plan(bundles, plan0, n_realizations=n_realizations)

    repairer = LocalRepair(bundles, ec, mc, num_reads=15, seed=seed)
    full = FullReoptimizer(bundles, ec, mc, seed=seed)

    rows = []
    local_times, full_times = [], []
    local_utils, full_utils = [], []
    recover_fracs = []
    per_rid_fail = {rid: 0 for rid in set(r for r, _ in plan0)}
    realization_outcomes = []
    for i in range(n_realizations):
        import random as _random
        sub = _random.Random(seed * 100000 + i)
        # reuse the engine's realization runner directly for the same seeds
        bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
        plan_objs = [(r, b, bmap[(r, b)]) for r, b in plan0]
        res = sim._run_realization(plan_objs, sub, None, {})
        failed = [rid for rid, out in res["outcomes"].items()
                  if out != "served"]
        realization_outcomes.append((i, failed))
        for rid in failed:
            per_rid_fail[rid] += 1

    for i, failed in realization_outcomes:
        committed = [(r, b) for r, b in plan0 if r not in failed]
        local_sel, t_local = repairer.repair(committed, failed)
        full_sel, t_full = full.reoptimize()
        u_local = _utility(local_sel, bundles)
        u_full = _utility(full_sel, bundles)
        # baseline with no recourse: the committed (survived) plan only
        u_no_repair = _utility(committed, bundles)
        local_times.append(t_local)
        full_times.append(t_full)
        local_utils.append(u_local)
        full_utils.append(u_full)
        if failed:
            recover_fracs.append(len([r for r in failed
                                      if r in {r2 for r2, _ in local_sel}])
                                  / len(failed))
        rows.append({
            "realization": i,
            "n_failed": len(failed),
            "n_local_recovered": len([r for r in failed
                                      if r in {r2 for r2, _ in local_sel}]),
            "t_local_s": t_local,
            "t_full_s": t_full,
            "u_local": u_local,
            "u_full": u_full,
            "u_no_repair": u_no_repair,
            "u_plan0": util0,
        })

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    recover_vs_norepair = []
    for r in rows:
        lost = r["u_plan0"] - r["u_no_repair"]
        if lost > 1e-9:
            recover_vs_norepair.append((r["u_local"] - r["u_no_repair"]) / lost)

    return {
        "n_requests": n_requests,
        "n_realizations": n_realizations,
        "plan0_utility": util0,
        "plan0_time_s": t_plan,
        "mean_failed_per_realization": _mean([len(f) for _, f in realization_outcomes]),
        "mean_t_local_s": _mean(local_times),
        "mean_t_full_s": _mean(full_times),
        "speedup": (_mean(full_times) / _mean(local_times)
                    if _mean(local_times) > 0 else float("inf")),
        "mean_u_local": _mean(local_utils),
        "mean_u_full": _mean(full_utils),
        "mean_recovery_rate": _mean(recover_fracs) if recover_fracs else 0.0,
        "lost_utility_recovered_frac": _mean(recover_vs_norepair) if recover_vs_norepair else 0.0,
        "per_request_fail_counts": per_rid_fail,
        "rows": rows,
    }
