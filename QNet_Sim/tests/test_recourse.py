"""Tests for adaptive recourse: local repair vs full reoptimization (Wave 3, item 3)."""

import math

import pytest

from simulation.recourse import (
    LocalRepair, FullReoptimizer, run_recourse_comparison,
)
from experiments.instances import (
    generate_chain_topology, generate_benchmark_instance,
    contention_sweep_instances,
)


def _topology(n_nodes=8, mem=10, raw=0.85):
    return generate_chain_topology(n_nodes=n_nodes, edge_capacity=6,
                                   memory_capacity=mem, raw_fidelity=raw,
                                   generation_prob=0.8)


def _bundles(n_req=4, seed=42):
    topo = _topology()
    inst = contention_sweep_instances(lambda: topo, [n_req], seed=seed)[
        "n%d" % n_req]
    return topo, inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]


def _feasible(selection, bundles, ec, mc):
    from optimization.qubo_optimizer import QUBOOptimizer
    opt = QUBOOptimizer(bundles, ec, mc)
    repaired = opt.repair_selection(list(selection))
    return set(repaired) == set(selection)


def test_local_repair_recovers_failed_requests():
    topo, bundles, ec, mc = _bundles()
    from optimization.metropolis_annealer import MetropolisAnnealer
    opt = MetropolisAnnealer(bundles, ec, mc, seed=42)
    plan0 = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                      max_iterations=2000, n_restarts=1,
                      steps_per_temperature=10)["selected"]
    assert plan0

    # fail the first request
    failed = [plan0[0][0]]
    committed = [(r, b) for r, b in plan0 if r not in failed]

    repairer = LocalRepair(bundles, ec, mc, num_reads=20, seed=42)
    sel, t = repairer.repair(committed, failed)
    assert isinstance(sel, list)
    # committed requests still served
    assert set(r for r, _ in committed) <= set(r for r, _ in sel)
    # repaired selection is capacity-feasible
    assert _feasible(sel, bundles, ec, mc)


def test_full_reoptimizer_is_feasible_and_deterministic():
    topo, bundles, ec, mc = _bundles()
    full = FullReoptimizer(bundles, ec, mc, seed=7)
    sel1, t1 = full.reoptimize()
    sel2, t2 = FullReoptimizer(bundles, ec, mc, seed=7).reoptimize()
    assert sel1 == sel2
    assert _feasible(sel1, bundles, ec, mc)


def test_recourse_comparison_schema_and_speedup():
    topo = _topology(n_nodes=8, mem=8)
    res = run_recourse_comparison(topo, n_requests=6, n_realizations=8,
                                  seed=42, tau_mem=50.0, swap_success=0.7)
    rows = res["rows"]
    assert len(rows) == 8
    assert res["plan0_utility"] > 0.0
    assert res["n_requests"] == 6
    for r in rows:
        assert 0 <= r["n_failed"] <= 6
        assert 0 <= r["n_local_recovered"] <= r["n_failed"]
        assert r["u_local"] >= r["u_no_repair"] - 1e-9
        assert r["t_local_s"] >= 0 and r["t_full_s"] > 0
    # local repair is dramatically faster on average
    assert res["mean_t_local_s"] < res["mean_t_full_s"]
    assert res["speedup"] >= 1.0
    # average recovery fraction within [0, 1]
    assert 0.0 <= res["mean_recovery_rate"] <= 1.0


def _stable_rows(rows):
    return sorted((r["realization"], r["n_failed"], r["n_local_recovered"],
                   r["u_local"], r["u_full"], r["u_no_repair"], r["u_plan0"])
                  for r in rows)


def test_recourse_comparison_deterministic():
    topo = _topology(n_nodes=6, mem=8)
    a = run_recourse_comparison(topo, n_requests=4, n_realizations=4, seed=1,
                                swap_success=0.7)
    b = run_recourse_comparison(topo, n_requests=4, n_realizations=4, seed=1,
                                swap_success=0.7)
    assert _stable_rows(a["rows"]) == _stable_rows(b["rows"])
    assert a["plan0_utility"] == b["plan0_utility"]
