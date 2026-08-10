"""Tests for optimality-gap certification and stochastic reliability (Wave 3, item 4)."""

import math

import pytest

from experiments.optimality_benchmark import (
    exact_ilp_solution, exact_brute_force, jain_index, run_gap_study,
    run_stochastic_reliability_benchmark,
)
from experiments.instances import (
    generate_chain_topology, generate_benchmark_instance,
)


def _small_instance(n_req=4, n_nodes=6, seed=3):
    topo = generate_chain_topology(n_nodes=n_nodes, edge_capacity=6,
                                   memory_capacity=10, raw_fidelity=0.85,
                                   generation_prob=0.8)
    import random as _r
    rng = _r.Random(seed)
    pairs = []
    for _ in range(n_req):
        s, d = rng.sample(topo["nodes"], 2)
        pairs.append((s, d, rng.uniform(10.0, 100.0), rng.uniform(0.5, 0.7)))
    return generate_benchmark_instance(topo, pairs, rng)


def test_ilp_matches_brute_force():
    bundles, ec, mc = _small_instance(n_req=4)
    ilp = exact_ilp_solution(bundles, ec, mc)
    bf = exact_brute_force(bundles, ec, mc)
    assert math.isclose(ilp["u_star"], bf["u_star"], rel_tol=1e-6)
    # the ILP selection is exactly feasible
    assert set(ilp["selected"]) == set(bf["selected"])


def test_ilp_selection_is_feasible():
    bundles, ec, mc = _small_instance(n_req=5)
    r = exact_ilp_solution(bundles, ec, mc)
    bmap = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    # one bundle per request
    rids = [rid for rid, _ in r["selected"]]
    assert len(rids) == len(set(rids))
    # capacity constraints respected
    edge_load, mem_load = {}, {}
    for rid, bid in r["selected"]:
        b = bmap[(rid, bid)]
        for e, d in b["edge_demands"].items():
            e = tuple(sorted(e))
            edge_load[e] = edge_load.get(e, 0) + d
        for n, d in b["memory_demands"].items():
            mem_load[n] = mem_load.get(n, 0) + d
    for e, load in edge_load.items():
        assert load <= ec.get(e, 0) or load <= ec.get(tuple(reversed(e)), 0)
    for n, load in mem_load.items():
        assert load <= mc[n]


def test_jain_index():
    assert math.isclose(jain_index([100.0, 100.0, 100.0]), 1.0, rel_tol=1e-9)
    assert math.isclose(jain_index([100.0, 50.0, 10.0]), 0.6772, rel_tol=1e-3)


def test_gap_study_rows_and_metropolis_quality():
    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                              memory_capacity=10,
                                              raw_fidelity=0.85,
                                              generation_prob=0.8)
    res = run_gap_study(topo_fn, sizes=[4, 6], n_instances=2, seed=42,
                        solvers=["hybrid", "metropolis"])
    rows = res
    assert len(rows) == 2 * 2 * 2
    for r in rows:
        assert r["gap_rel"] >= -1e-12
        assert r["t_exact_s"] > 0
        assert r["u_star"] >= r["u_solver"] - 1e-9
    metro = [r for r in rows if r["solver"] == "metropolis"]
    # the strong annealer stays near the exact optimum on these chains
    assert all(r["gap_rel"] < 0.15 for r in metro)


def test_stochastic_reliability_benchmark():
    topo = generate_chain_topology(n_nodes=6, edge_capacity=6,
                                   memory_capacity=10, raw_fidelity=0.85,
                                   generation_prob=0.8)
    res = run_stochastic_reliability_benchmark(
        topo, n_requests=5, n_realizations=15, seed=42,
        solvers=["metropolis"])
    rows = res["rows"]
    assert len(rows) == 1
    r = rows[0]
    assert r["solver"] == "metropolis"
    assert r["e_utility"] >= 0.0
    assert r["var_utility"] >= 0.0
    assert 0.0 <= r["served_ratio"] <= 1.0
    assert 0.0 <= r["sla_violation_prob"] <= 1.0
    assert 0.0 <= r["mean_jain_index"] <= 1.0
    assert r["param_expected_utility"] >= 0.0
