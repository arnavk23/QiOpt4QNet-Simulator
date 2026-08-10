"""Tests for purification as an optimization variable (Wave 3, item 2)."""

import pytest

from optimization.purification_scheduler import (
    purification_tradeoff, purification_frontier, run_purification_comparison,
    run_purification_sweep,
)
from experiments.instances import generate_chain_topology


def _topology(n_nodes=7, raw=0.85, g=0.8):
    return generate_chain_topology(n_nodes=n_nodes, edge_capacity=6,
                                   memory_capacity=10, raw_fidelity=raw,
                                   generation_prob=g)


def test_tradeoff_feasibility_and_monotonicity():
    topo = _topology()
    rows = purification_tradeoff(topo, topo["nodes"], min_fidelity=0.6,
                                 weight=50.0, q_max=4)
    assert rows
    # only feasible levels are returned (meet the fidelity floor)
    for r in rows:
        assert r["fidelity"] >= 0.6 - 1e-12
    # fidelity and cost are non-decreasing in k
    fids = [r["fidelity"] for r in rows]
    costs = [r["bell_pair_cost"] for r in rows]
    assert fids == sorted(fids)
    assert costs == sorted(costs)


def test_tradeoff_empty_when_unfeasible():
    topo = _topology(raw=0.6)
    rows = purification_tradeoff(topo, topo["nodes"], min_fidelity=0.95,
                                 weight=50.0, q_max=4)
    assert rows == []


def test_frontier_no_dominance():
    topo = _topology()
    frontier = purification_frontier(topo, topo["nodes"], min_fidelity=0.6,
                                     weight=50.0, q_max=4)
    assert frontier
    for p in frontier:
        for q in frontier:
            if p["k"] == q["k"]:
                continue
            better = all([
                q["fidelity"] >= p["fidelity"] - 1e-12,
                q["success_probability"] >= p["success_probability"] - 1e-12,
                q["latency"] <= p["latency"] + 1e-12,
                q["bell_pair_cost"] <= p["bell_pair_cost"] + 1e-12,
                q["memory_demand"] <= p["memory_demand"] + 1e-12,
            ])
            assert not better, f"level k={q['k']} dominates k={p['k']}"


def test_purification_aware_serves_at_least_agnostic():
    topo = _topology(n_nodes=7)
    res = run_purification_comparison(topo, n_requests=8, q_max=4, seed=42)
    rows = {r["regime"]: r for r in res["rows"]}
    aware = rows["purification_aware"]
    agnostic = rows["purification_agnostic"]
    assert aware["served"] >= agnostic["served"]
    # offering more purification levels can only add feasible bundles
    assert aware["n_bundles"] > agnostic["n_bundles"]


def test_purification_sweep_requires_k_for_long_paths():
    rows = run_purification_sweep(path_lengths=[2, 4, 6, 8],
                                  min_fidelity_values=[0.6], raw_fidelity=0.85,
                                  generation_prob=0.8)
    assert rows
    for n_links in [2, 4, 6, 8]:
        k_here = {r["k"] for r in rows if r["path_length"] == n_links}
        assert k_here, f"no feasible purification level for {n_links} links"
    # raw-fidelity chain: k=0 only feasible on short paths
    k0_paths = {r["path_length"] for r in rows if r["k"] == 0}
    k4_paths = {r["path_length"] for r in rows if r["k"] == 4}
    assert max(k0_paths, default=0) < max(k4_paths, default=0)
