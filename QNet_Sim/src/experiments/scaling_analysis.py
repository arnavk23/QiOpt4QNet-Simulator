import csv
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from experiments.instances import contention_sweep_instances
from experiments.benchmark import build_metropolis, build_tensor_network

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiments"))
os.makedirs(OUT, exist_ok=True)

try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _utility(bundles, selected):
    util_of = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return sum(util_of.get(k, 0.0) for k in selected)


def _served_ratio(selected, n_requests):
    return len(selected) / max(n_requests, 1)


def run_quality_time_curves(topology_fn: Callable,
                            n_requests_list: Optional[List[int]] = None,
                            bond_dims: Optional[List[int]] = None,
                            metropolis_budgets: Optional[List[int]] = None,
                            seed: int = 42,
                            topology_name: str = "chain") -> List[Dict]:
    """Run both solvers over a grid of (request count, time budget/bond
    dimension) and record achieved quality vs wall-clock time.

    The Metropolis annealer is swept over iteration budgets (its wall time is
    proportional to the budget), the tensor-network solver over bond dimension
    chi (its wall time scales like N*chi^3).  This yields the quality-vs-time
    curve Q(t; N) for each solver that the scaling-law fits below consume.
    """
    if n_requests_list is None:
        n_requests_list = [4, 8, 16, 24]
    if bond_dims is None:
        bond_dims = [2, 4, 8, 16, 32]
    if metropolis_budgets is None:
        metropolis_budgets = [500, 1000, 2000, 5000]

    instances = contention_sweep_instances(topology_fn, n_requests_list, seed=seed)
    rows: List[Dict] = []

    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

        for budget in metropolis_budgets:
            sf = build_metropolis(b, ec, mc, seed=seed)
            t0 = time.perf_counter()
            r = sf(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                   max_iterations=budget, cooling_rate=0.97, n_restarts=1,
                   steps_per_temperature=10, initial_temperature=100.0,
                   congestion_penalty=0.0, memory_congestion_penalty=0.0)
            t = time.perf_counter() - t0
            rows.append({
                "topology": topology_name,
                "n_requests": n_req,
                "solver": "Metropolis",
                "budget": budget,
                "bond_dim": -1,
                "time_s": t,
                "served": len(r.get("selected", [])),
                "served_ratio": _served_ratio(r.get("selected", []), n_req),
                "utility": _utility(b, r.get("selected", [])),
            })

        for bd in bond_dims:
            sf = build_tensor_network(b, ec, mc)
            t0 = time.perf_counter()
            r = sf(bond_dim=bd, beta=5.0, edge_penalty=10.0, memory_penalty=10.0,
                   max_sweeps=10, congestion_penalty=0.0, memory_congestion_penalty=0.0)
            t = time.perf_counter() - t0
            rows.append({
                "topology": topology_name,
                "n_requests": n_req,
                "solver": "TensorNetwork",
                "budget": -1,
                "bond_dim": bd,
                "time_s": t,
                "served": len(r.get("selected", [])),
                "served_ratio": _served_ratio(r.get("selected", []), n_req),
                "utility": _utility(b, r.get("selected", [])),
            })

    _write("scaling_curves.csv", rows)
    return rows


def _write(path: str, rows: List[Dict]):
    path = os.path.join(OUT, path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def _metropolis_model(t, q_opt, c1, c2, n):
    """Q(t; N) ~ Q_opt - c1*N*exp(-t/(c2*N))  (mixing-limited scaling)."""
    return q_opt - c1 * n * np.exp(-t / (c2 * n))


def fit_metropolis_law(rows: List[Dict]) -> Dict:
    """Fit Q(t; N) = Q_opt - c1*N*exp(-t/(c2*N)) to the Metropolis rows.

    Q_opt is the best utility observed across all solvers at each N (an upper
    anchor), and c1, c2 are shared across N so the law predicts the full curve.
    """
    if not _HAS_SCIPY:
        return {"error": "scipy not available"}

    best_opt = {}
    for r in rows:
        n = int(r["n_requests"])
        best_opt[n] = max(best_opt.get(n, 0.0), float(r["utility"]))

    meta = [r for r in rows if r["solver"] == "Metropolis"]
    t_all, q_all, n_all = [], [], []
    for r in meta:
        n = int(r["n_requests"])
        if n in best_opt:
            t_all.append(float(r["time_s"]))
            q_all.append(float(r["utility"]))
            n_all.append(n)

    t_all, q_all, n_all = map(np.array, (t_all, q_all, n_all))

    if np.var(q_all) < 1e-12:
        # Quality is independent of time budget on every tested instance: both
        # solvers saturate at the same utility within the smallest budget.  The
        # fitted time constants are then only upper bounds.
        return {"model": "Q(t,N) = Q_opt - c1*N*exp(-t/(c2*N))",
                "q_opt": float(np.mean(q_all)),
                "c1": 0.0, "c2": None, "r2": None,
                "n_points": len(t_all),
                "note": "quality saturated at all time budgets (c1->0, c2 undefined); "
                        "observed quality-vs-time law is flat"}

    def model(params, t, n):
        q_opt, c1, c2 = params
        return q_opt - c1 * n * np.exp(-t / (c2 * n))

    def resid(params):
        return model(params, t_all, n_all) - q_all

    q_opt0 = max(best_opt.values())
    from scipy.optimize import least_squares
    result = least_squares(resid, x0=[q_opt0, 1.0, 0.5], bounds=([q_opt0 * 0.5, 0.0, 1e-3],
                                                                  [q_opt0 * 2.0, np.inf, np.inf]))
    q_opt, c1, c2 = result.x
    fitted = model([q_opt, c1, c2], t_all, n_all)
    r2 = 1.0 - float(np.sum((q_all - fitted) ** 2) / np.sum((q_all - q_all.mean()) ** 2))
    return {"model": "Q(t,N) = Q_opt - c1*N*exp(-t/(c2*N))",
            "q_opt": float(q_opt), "c1": float(c1), "c2": float(c2),
            "r2": r2, "n_points": len(t_all)}


def fit_mps_law(rows: List[Dict]) -> Dict:
    """Fit Q(N; chi) = Q_opt - c3(N)*exp(-c4*chi) and runtime t = c5*N*chi^3.

    For each request count N we fit the truncation-limited quality curve in chi,
    and we regress the observed runtime against N*chi^3 to confirm the sweep cost.
    """
    if not _HAS_SCIPY:
        return {"error": "scipy not available"}

    best_opt = {}
    for r in rows:
        n = int(r["n_requests"])
        best_opt[n] = max(best_opt.get(n, 0.0), float(r["utility"]))

    tn = [r for r in rows if r["solver"] == "TensorNetwork"]
    per_n = {}
    for r in tn:
        per_n.setdefault(int(r["n_requests"]), []).append(r)

    fits = {}
    for n, rs in per_n.items():
        chi = np.array([float(r["bond_dim"]) for r in rs], dtype=float)
        q = np.array([float(r["utility"]) for r in rs], dtype=float)
        t = np.array([float(r["time_s"]) for r in rs], dtype=float)
        if len(chi) < 3:
            continue

        c5 = float(np.mean(t / (np.maximum(n, 1) * chi ** 3))) if np.any(chi) else 0.0
        q_opt = best_opt.get(n, q.max())

        if np.all(q == q[0]):
            # Quality saturates at the smallest tested chi: the truncation
            # limit c3 is (effectively) zero for this instance size.  The
            # runtime law t ~ c5*N*chi^3 is still well-defined.
            fits[str(n)] = {"q_opt": float(q_opt), "c3": 0.0, "c4": None,
                            "c5": c5, "r2": None,
                            "note": "quality saturated at chi_min (c3 -> 0)"}
            continue

        c4_lo, c4_hi = 0.0, 2.0

        def fn(chi, c3, c4):
            return q_opt - c3 * np.exp(-c4 * (chi - 1))

        try:
            popt, _ = curve_fit(fn, chi, q, p0=[(q_opt - q.min()) / np.exp(-0.5 * (chi[0] - 1)), 0.5],
                                bounds=([0.0, c4_lo], [q_opt * 4.0, c4_hi]), maxfev=5000)
        except Exception:
            continue
        c3, c4 = popt
        pred = fn(chi, c3, c4)
        r2 = 1.0 - float(np.sum((q - pred) ** 2) / np.sum((q - q.mean()) ** 2))

        fits[str(n)] = {"q_opt": float(q_opt), "c3": float(c3), "c4": float(c4),
                        "c5": c5, "r2": r2}

    return {"model": "Q(N;chi) = Q_opt - c3(N)*exp(-c4*chi), t ~ c5*N*chi^3",
            "per_n": fits}


def find_crossover(rows: List[Dict], threshold_fraction: float = 0.9) -> Dict:
    """Estimate the crossover request count N*(chi).

    For each (N, chi) we compute the time at which each solver reaches
    ``threshold_fraction * Q_opt`` by interpolating its quality-vs-time curve
    (Metropolis: quality vs iteration budget; tensor network: quality vs chi,
    with runtime t ~ c5*N*chi^3).  The crossover N*(chi) is the smallest N at
    which the tensor-network solver reaches the threshold faster than the
    Metropolis annealer.
    """
    if not _HAS_SCIPY:
        return {"error": "scipy not available"}

    best_opt = {}
    for r in rows:
        n = int(r["n_requests"])
        best_opt[n] = max(best_opt.get(n, 0.0), float(r["utility"]))
    if not best_opt:
        return {}

    meta_by_n = {}
    for r in rows:
        if r["solver"] == "Metropolis":
            meta_by_n.setdefault(int(r["n_requests"]), []).append(r)
    tn_by_n = {}
    for r in rows:
        if r["solver"] == "TensorNetwork":
            tn_by_n.setdefault(int(r["n_requests"]), []).append(r)

    ns = sorted(set(meta_by_n) & set(tn_by_n))
    summary = []
    crossover = None
    for n in ns:
        target = threshold_fraction * best_opt[n]
        m_rows = sorted(meta_by_n[n], key=lambda r: float(r["budget"]))
        t_m = _interp_time_to_quality(m_rows, target)
        for r in tn_by_n[n]:
            chi = int(r["bond_dim"])
            t_tn = _mps_time_to_quality(tn_by_n[n], chi, target, n)
            summary.append({"n_requests": n, "bond_dim": chi,
                            "threshold": threshold_fraction,
                            "metropolis_time_s": t_m,
                            "tn_time_s": t_tn,
                            "tn_faster": t_tn < t_m if t_m is not None and t_tn is not None else None})
            if t_m is not None and t_tn is not None and t_tn < t_m:
                if crossover is None or n < crossover["n_requests"]:
                    crossover = {"n_requests": n, "bond_dim": chi,
                                 "metropolis_time_s": t_m, "tn_time_s": t_tn}

    return {"threshold_fraction": threshold_fraction,
            "crossover": crossover, "rows": summary}


def _interp_time_to_quality(rows: List[Dict], target: float) -> Optional[float]:
    pts = sorted((float(r["time_s"]), float(r["utility"])) for r in rows)
    if pts[0][1] >= target:
        return pts[0][0]
    for (t0, q0), (t1, q1) in zip(pts, pts[1:]):
        if q1 >= target:
            if q1 == q0:
                return t1
            frac = (target - q0) / (q1 - q0)
            return t0 + frac * (t1 - t0)
    return None


def _mps_time_to_quality(rows: List[Dict], chi: int, target: float,
                         n_requests: int) -> Optional[float]:
    """Interpolate the TN quality-vs-chi curve to find the runtime at which it
    first reaches ``target`` (runtime model t ~ c5*N*chi^3)."""
    pts = sorted((int(r["bond_dim"]), float(r["utility"])) for r in rows)
    if not pts:
        return None
    if pts[0][1] >= target:
        # already at (or above) target at the smallest chi: report its runtime
        chi_needed = pts[0][0]
    else:
        chi_needed = None
        for (c0, q0), (c1, q1) in zip(pts, pts[1:]):
            if q1 >= target:
                if q1 == q0:
                    chi_needed = c1
                else:
                    chi_needed = c0 + (target - q0) / (q1 - q0) * (c1 - c0)
                break
    if chi_needed is None:
        return None
    c5 = float(np.mean([float(r["time_s"]) / (max(n_requests, 1) * max(int(r["bond_dim"]), 1) ** 3)
                        for r in rows]))
    return c5 * max(n_requests, 1) * max(chi_needed, 1) ** 3


def analyze(topology_fn: Callable, n_requests_list: Optional[List[int]] = None,
            threshold_fraction: float = 0.9) -> Dict:
    """Run the curves and return the fitted scaling laws plus crossover."""
    rows = run_quality_time_curves(topology_fn, n_requests_list=n_requests_list)
    meta_fit = fit_metropolis_law(rows)
    mps_fit = fit_mps_law(rows)
    cross = find_crossover(rows, threshold_fraction)
    return {"metropolis": meta_fit, "mps": mps_fit, "crossover": cross}


def main():
    print("=" * 60)
    print("  Scaling-law / quality-time tradeoff analysis")
    print("=" * 60)

    from experiments.instances import generate_grid_topology
    topo = lambda: generate_grid_topology(rows=4, cols=4, edge_capacity=5,
                                          latency=5.0)

    print("\nCollecting quality-time curves (Metropolis budgets, TN chi sweep)...")
    rows = run_quality_time_curves(topo, n_requests_list=[8, 16, 24],
                                   topology_name="grid_4x4")

    print("\nFitting Metropolis scaling law...")
    m = fit_metropolis_law(rows)
    print(m)

    print("\nFitting MPS scaling law...")
    p = fit_mps_law(rows)
    print(p)

    print("\nCrossover analysis (threshold=0.9*Q_opt)...")
    c = find_crossover(rows, 0.9)
    print("crossover:", c.get("crossover"))

    summary = {"metropolis": m, "mps": p, "crossover": c}
    with open(os.path.join(OUT, "scaling_law_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT}/scaling_law_summary.json")


if __name__ == "__main__":
    main()
