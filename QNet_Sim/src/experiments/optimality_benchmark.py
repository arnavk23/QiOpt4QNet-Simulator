"""Optimality-gap certification and stochastic reliability evaluation (Wave 3, item 4).

Two independent upgrades to the evaluation framework:

**Optimality-gap certification.**  For small instances the bundle-selection
problem is solved *exactly* with a mixed-integer linear program (HiGHS via
``scipy.optimize.milp``), giving :math:`U^*`.  Every approximate solver's
output is then certified as a gap

.. math::

    Gap = \\frac{U^* - U_{solver}}{|U^*|}

over hundreds of instances, producing the *solution-quality vs. runtime*
curves that let us claim, e.g., "the hybrid solver achieves a ≤ X% mean
optimality gap while using ≤ Y% of the exact solver's time."

**Stochastic reliability evaluation.**  A solver's plan is executed by the
discrete-event engine over ``N`` realizations, and the *parametric* expected
utility (``sum bundle.utility``) is confronted with the *sampled* statistics:
``E[U]``, ``Var[U]``, the delivered-fidelity distribution (5th/50th/95th
percentiles), the SLA-violation probability ``P(F < F_min)``, and Jain's
fairness index over served requests.  The gap between ``param_expected`` and
``E[U_sampled]`` is the *reliability gap* that deterministic models hide.
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.instances import generate_benchmark_instance, _shortest_paths


# ----------------------------------------------------------------------
# exact optimum
# ----------------------------------------------------------------------
def exact_ilp_solution(bundles: List[dict], edge_capacities: dict,
                       memory_capacities: dict,
                       time_limit: Optional[float] = 60.0) -> Dict:
    """Exact optimum of the bundle-selection ILP.

    maximize   sum_i u_i x_i
    subject to sum_{i in req} x_i <= 1            (one bundle per request)
               sum_i d_{e,i} x_i <= cap_e         (edge capacity)
               sum_i d_{n,i} x_i <= cap_n         (memory capacity)
               x_i in {0, 1}

    Solved with HiGHS (``scipy.optimize.milp``).  Returns the optimum
    utility, the selection, the solver status and wall time.
    """
    n = len(bundles)
    rows_A = []
    rows_ub = []
    rows_lb = []

    by_request: Dict[str, List[int]] = {}
    for i, b in enumerate(bundles):
        by_request.setdefault(b["request_id"], []).append(i)
    for idxs in by_request.values():
        row = np.zeros(n)
        row[idxs] = 1.0
        rows_A.append(row)
        rows_ub.append(1.0)
        rows_lb.append(-np.inf)

    for e, cap in edge_capacities.items():
        row = np.zeros(n)
        for i, b in enumerate(bundles):
            row[i] = b["edge_demands"].get(e, b["edge_demands"].get(
                tuple(sorted(e)), 0))
        rows_A.append(row)
        rows_ub.append(float(cap))
        rows_lb.append(-np.inf)

    for node, cap in memory_capacities.items():
        row = np.zeros(n)
        for i, b in enumerate(bundles):
            row[i] = b["memory_demands"].get(node, 0)
        rows_A.append(row)
        rows_ub.append(float(cap))
        rows_lb.append(-np.inf)

    from scipy.optimize import Bounds, LinearConstraint, milp

    c = -np.array([b["utility"] for b in bundles], dtype=float)
    A = np.vstack(rows_A)
    constraints = LinearConstraint(A, np.array(rows_lb), np.array(rows_ub))
    t0 = time.perf_counter()
    res = milp(c=c, constraints=constraints,
               integrality=np.ones(n), bounds=Bounds(0.0, 1.0),
               options={"time_limit": time_limit})
    elapsed = time.perf_counter() - t0

    if res.status in (0, 1) and res.x is not None:
        selected = [(bundles[i]["request_id"], bundles[i]["bundle_id"])
                    for i in range(n) if res.x[i] > 0.5]
        u_star = float(-res.fun)
    else:
        selected = []
        u_star = None

    return {
        "u_star": u_star,
        "selected": selected,
        "status": res.status,
        "wall_time_s": elapsed,
        "n_variables": n,
    }


def exact_brute_force(bundles: List[dict], edge_capacities: dict,
                      memory_capacities: dict) -> Dict:
    """Exact optimum by exhaustive enumeration (test / tiny-instance fallback)."""
    from itertools import product

    by_request: Dict[str, List[dict]] = {}
    for b in bundles:
        by_request.setdefault(b["request_id"], []).append(b)
    req_ids = list(by_request)
    options = [[(b["request_id"], b["bundle_id"]) for b in by_request[rid]]
               + [None] for rid in req_ids]
    best_u, best_sel = None, []
    for combo in product(*options):
        sel = [k for k in combo if k is not None]
        e_load = {}
        m_load = {}
        u = 0.0
        bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
        feasible = True
        for rid, bid in sel:
            b = bmap[(rid, bid)]
            u += b["utility"]
            for e, d in b["edge_demands"].items():
                e_load[e] = e_load.get(e, 0) + d
                if e_load[e] > edge_capacities.get(e, 0):
                    feasible = False
                    break
            if not feasible:
                break
            for nd, d in b["memory_demands"].items():
                m_load[nd] = m_load.get(nd, 0) + d
                if m_load[nd] > memory_capacities.get(nd, 0):
                    feasible = False
                    break
        if feasible and (best_u is None or u > best_u):
            best_u, best_sel = u, sel
    return {"u_star": best_u, "selected": best_sel, "status": 0,
            "wall_time_s": 0.0, "n_variables": len(bundles)}


# ----------------------------------------------------------------------
# gap study
# ----------------------------------------------------------------------
def _build_instance(topology: dict, n_requests: int, seed: int,
                    fidelity_bounds=(0.5, 0.8)):
    rng = random.Random(seed)
    nodes = topology["nodes"]
    pairs = []
    for _ in range(n_requests):
        src, dst = rng.sample(nodes, 2)
        pairs.append((src, dst, rng.uniform(10.0, 100.0),
                      rng.uniform(*fidelity_bounds)))
    bundles, ec, mc = generate_benchmark_instance(topology, pairs, rng)
    # stamp metadata needed by the stochastic evaluator / fairness
    meta = {(src, dst): (w, mf) for src, dst, w, mf in pairs}
    for b in bundles:
        src, dst = b["path"][0], b["path"][-1]
        w, mf = meta[(src, dst)]
        b["min_fidelity"] = mf
        b["weight"] = w
    return bundles, ec, mc


def _utility(selection: List[Tuple[str, str]], bundles: List[dict]) -> float:
    bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    return sum(bmap.get(k, {"utility": 0.0})["utility"] for k in selection)


def _run_solver(name: str, bundles: List[dict], ec: dict, mc: dict,
                seed: int) -> Tuple[List[Tuple[str, str]], float, Dict]:
    """Run one approximate solver; returns (selection, wall_time, extra)."""
    t0 = time.perf_counter()
    if name == "metropolis":
        from optimization.metropolis_annealer import MetropolisAnnealer
        opt = MetropolisAnnealer(bundles, ec, mc, seed=seed)
        r = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        sel = r.get("selected", [])
    elif name == "hybrid":
        from optimization.adaptive_qubo import adaptive_qubo_solve
        r = adaptive_qubo_solve(bundles, ec, mc, k=8, num_reads=20, seed=seed)
        sel = r.get("selected", [])
    elif name == "qubo_full":
        from optimization.adaptive_qubo import reference_solution
        r = reference_solution(bundles, ec, mc, num_reads=20, seed=seed)
        sel = r.get("selected", [])
    elif name == "sqa":
        from optimization.qubo_optimizer import QUBOOptimizer
        from optimization.openjij_solver import solve_sqa
        opt = QUBOOptimizer(bundles, ec, mc)
        bqm = opt.to_bqm(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                         congestion_penalty=0.0, memory_congestion_penalty=0.0)
        resp = solve_sqa(bqm, num_reads=20, seed=seed)
        sel = opt.decode_sample(resp.first.sample, repair=True)
    else:
        raise ValueError(f"unknown solver {name!r}")
    elapsed = time.perf_counter() - t0
    return sel, elapsed, {}


def run_gap_study(topology_fn, sizes: Optional[List[int]] = None,
                  n_instances: int = 12, seed: int = 42,
                  solvers: Optional[List[str]] = None,
                  time_limit: float = 60.0) -> List[Dict]:
    """Certify solver selections against the exact ILP optimum.

    Returns one row per (size, instance, solver): ``u_star``, solver utility,
    absolute and relative gap, and both wall times.
    """
    if sizes is None:
        sizes = [4, 6, 8, 10]
    if solvers is None:
        solvers = ["hybrid", "metropolis", "qubo_full", "sqa"]
    rows = []
    for n_req in sizes:
        for inst in range(n_instances):
            iseed = seed * 1000 + n_req * 100 + inst
            bundles, ec, mc = _build_instance(topology_fn(), n_req, iseed)
            exact = exact_ilp_solution(bundles, ec, mc, time_limit=time_limit)
            u_star = exact["u_star"]
            if u_star is None:
                continue
            for solver in solvers:
                try:
                    sel, t_solver, _ = _run_solver(solver, bundles, ec, mc, seed=iseed)
                except Exception:
                    continue
                u_solver = _utility(sel, bundles)
                gap_abs = u_star - u_solver
                rows.append({
                    "n_requests": n_req,
                    "n_bundles": len(bundles),
                    "n_variables": exact["n_variables"],
                    "solver": solver,
                    "instance": inst,
                    "u_star": u_star,
                    "u_solver": u_solver,
                    "gap_abs": gap_abs,
                    "gap_rel": gap_abs / max(abs(u_star), 1e-12),
                    "t_exact_s": exact["wall_time_s"],
                    "t_solver_s": t_solver,
                })
    return rows


# ----------------------------------------------------------------------
# stochastic reliability evaluation
# ----------------------------------------------------------------------
def jain_index(values: List[float]) -> float:
    """Jain's fairness index over a list of per-request utilities."""
    if not values:
        return 1.0
    total = sum(values)
    if total <= 0:
        return 1.0
    n = len(values)
    return total ** 2 / (n * sum(v * v for v in values))


def run_stochastic_reliability_benchmark(topology: dict, n_requests: int = 8,
                                         n_realizations: int = 60,
                                         seed: int = 42,
                                         tau_mem: float = 50.0,
                                         swap_success: float = 0.95,
                                         solvers: Optional[List[str]] = None,
                                         ) -> Dict:
    """Execute each solver's plan in the DES and compare parametric vs sampled.

    Rows carry ``param_expected_utility``, sampled ``e_utility`` / ``var`` /
    ``std``, served ratio, fidelity quantiles, ``sla_violation_prob`` and the
    mean Jain fairness index over delivered requests.
    """
    if solvers is None:
        solvers = ["hybrid", "metropolis", "sqa"]
    from simulation.discrete_event_engine import StochasticEventSimulator

    bundles, ec, mc = _build_instance(topology, n_requests, seed)
    sim = StochasticEventSimulator(topology, tau_mem=tau_mem,
                                   swap_success=swap_success, seed=seed)
    sla = {b["request_id"]: b["min_fidelity"] for b in bundles}

    rows = []
    for solver in solvers:
        try:
            sel, _t, _ = _run_solver(solver, bundles, ec, mc, seed=seed)
        except Exception as e:
            rows.append({"solver": solver, "n_requests": n_requests,
                         "error": str(e)})
            continue
        res = sim.simulate_plan(bundles, sel, n_realizations=n_realizations,
                                sla_thresholds=sla)
        # per-realization Jain index over delivered per-request utilities
        jains = []
        bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
        plan_objs = [(r, b, bmap[(r, b)]) for r, b in sel]
        import random as _random
        for i in range(n_realizations):
            sub = _random.Random(seed * 100000 + i)
            rr = sim._run_realization(plan_objs, sub, None, sla)
            contribs = []
            for rid, _f, _l, _c, _w, _fm in rr["delivered"]:
                b = bmap[(rid, next(bid for r2, bid in sel if r2 == rid))]
                contribs.append(b["utility"] / max(b.get("success_probability", 1e-12), 1e-12))
            jains.append(jain_index(contribs))
        rows.append({
            "solver": solver,
            "n_requests": n_requests,
            "n_selected": len(sel),
            "param_expected_utility": res["param_expected_utility"],
            "e_utility": res["e_utility"],
            "var_utility": res["var_utility"],
            "std_utility": res["std_utility"],
            "reliability_gap": res["utility_gap"],
            "e_served": res["e_served"],
            "served_ratio": res["served_ratio"],
            "e_delivered_fidelity": res["e_delivered_fidelity"],
            "fid_q05": res["fid_q05"],
            "fid_q50": res["fid_q50"],
            "fid_q95": res["fid_q95"],
            "sla_violation_prob": res["sla_violation_prob"],
            "mean_jain_index": sum(jains) / len(jains) if jains else 1.0,
            "failure_causes": res["failure_causes"],
        })
    return {"n_requests": n_requests, "n_realizations": n_realizations,
            "rows": rows}
