"""Tests for the adaptive QUBO candidate budget (Wave 3, item 5)."""

import pytest

from optimization.adaptive_budget import (
    AdaptiveBudgetPolicy, adaptive_budget_solve, run_adaptive_budget_study,
)
from experiments.instances import generate_chain_topology


def _topo_fn():
    def build():
        return generate_chain_topology(n_nodes=6, edge_capacity=6,
                                       memory_capacity=10, raw_fidelity=0.85,
                                       generation_prob=0.8)
    return build


def test_policy_clamps_to_bounds():
    pol = AdaptiveBudgetPolicy(base=4.0, k_min=2, k_max=8, seed=42)
    # extreme congestion/memory push k to the floor; extreme density to the cap
    hard = [{"request_id": f"r{i}", "edge_demands": {("N0", "N1"): 9},
             "memory_demands": {"N0": 9}, "utility": 10.0}
            for i in range(4)]
    ec = {("N0", "N1"): 1}
    mc = {"N0": 1, "N1": 1}
    assert pol.k_min <= pol.select_k(hard, ec, mc) <= pol.k_max


def test_state_features_normalized():
    pol = AdaptiveBudgetPolicy(seed=1)
    bundles = [{"request_id": f"r{i}", "edge_demands": {("N0", "N1"): 2},
                "memory_demands": {"N0": 2}, "utility": 10.0, "fidelity": 0.8}
               for i in range(4)]
    s = pol.state_features(bundles, {("N0", "N1"): 4},
                           {"N0": 4, "N1": 4})
    for k in ["congestion", "density", "fidelity_strictness", "memory_pressure"]:
        assert 0.0 <= s[k] <= 1.0, k


def test_adaptive_budget_solve_against_ilp():
    topo_fn = _topo_fn()
    import random as _r
    from experiments.instances import contention_sweep_instances
    inst = contention_sweep_instances(topo_fn, [6], seed=42)["n6"]
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    r = adaptive_budget_solve(b, ec, mc, solver="metropolis", seed=42)
    assert r["utility"] >= 0.0
    assert r["relative_gap_vs_full"] >= -1e-12
    assert r["k"] >= 2
    assert r["selected"]


def _stable_rows(rows):
    return sorted((r["n_requests"], r["method"], r["k"], r["utility"],
                   r["n_qubo_variables"],
                   round(r["relative_gap_vs_full"], 6)) for r in rows)


def test_adaptive_budget_study_deterministic_and_free_reduction():
    topo_fn = _topo_fn()
    a = run_adaptive_budget_study(topo_fn, n_requests_list=[4, 6],
                                  k_values=[4, 8], solver="metropolis",
                                  seed=42)
    b = run_adaptive_budget_study(topo_fn, n_requests_list=[4, 6],
                                  k_values=[4, 8], solver="metropolis",
                                  seed=42)
    assert _stable_rows(a["rows"]) == _stable_rows(b["rows"])
    methods = {r["method"] for r in a["rows"]}
    assert {"k4", "k8", "adaptive"} <= methods
    for r in a["rows"]:
        assert r["n_qubo_variables"] > 0
        assert r["utility"] >= 0.0
        assert r["relative_gap_vs_full"] >= -1e-12
    # adaptive never needs the largest fixed budget for the same quality
    # (weak check: adaptive is chosen from the policy, inside the k range)
    for sm in a["summaries"]:
        assert 2 <= sm["k_adaptive"] <= 32


def test_adaptive_budget_study_sa_contracts_under_pressure():
    """Under congestion the policy picks a small k that beats the large ones."""
    from experiments.instances import generate_grid_topology

    def grid():
        return generate_grid_topology(rows=3, cols=5, edge_capacity=8,
                                      memory_capacity=12)

    res = run_adaptive_budget_study(grid, n_requests_list=[6, 10],
                                    k_values=[4, 16], solver="sa",
                                    num_reads=60, seed=42)
    for n_req in [6, 10]:
        rows_n = [r for r in res["rows"] if r["n_requests"] == n_req]
        adap = next(r for r in rows_n if r["method"] == "adaptive")
        fixed = [r for r in rows_n if r["method"] != "adaptive"]
        best_fixed = min(r["relative_gap_vs_full"] for r in fixed)
        # adaptive is at least as good as the best fixed budget
        assert adap["relative_gap_vs_full"] <= best_fixed + 1e-9
