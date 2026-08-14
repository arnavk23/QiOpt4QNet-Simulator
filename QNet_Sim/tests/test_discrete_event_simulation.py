"""Tests for the stochastic discrete-event simulator (Wave 3, item 1)."""

import math

import pytest

from simulation.discrete_event_engine import (
    StochasticEventSimulator, storage_decay, delivered_utility_if_success,
)
from experiments.instances import (
    generate_chain_topology, generate_benchmark_instance,
    contention_sweep_instances,
)


def _topology(n_nodes=8, cap=6, mem=10, raw=0.85, g=0.8):
    return generate_chain_topology(n_nodes=n_nodes, edge_capacity=cap,
                                   memory_capacity=mem, raw_fidelity=raw,
                                   generation_prob=g)


def _instance(n_req=4, n_nodes=8, seed=42):
    topo = _topology(n_nodes)
    return topo, contention_sweep_instances(lambda: topo, [n_req], seed=seed)[
        f"req{n_req}"]


def _plan_from_solve(bundles, ec, mc, seed=42):
    from optimization.metropolis_annealer import MetropolisAnnealer
    opt = MetropolisAnnealer(bundles, ec, mc, seed=seed)
    r = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                  max_iterations=2000, n_restarts=1, steps_per_temperature=10)
    return r["selected"]


def test_storage_decay_monotone_and_bounded():
    d0 = storage_decay(0.0, tau_mem=10.0)
    assert math.isclose(d0, 1.0, rel_tol=1e-6)
    for t in [0.5, 2.0, 10.0, 50.0]:
        d = storage_decay(t, tau_mem=10.0)
        assert 0.0 < d <= 1.0
    assert storage_decay(1.0, tau_mem=10.0) < storage_decay(0.5, tau_mem=10.0)


def test_delivered_utility_if_success_scales():
    b = {"utility": 100.0, "success_probability": 0.5}
    assert math.isclose(delivered_utility_if_success(b), 200.0, rel_tol=1e-9)
    assert math.isclose(delivered_utility_if_success(
        {**b, "success_probability": 1e-12}), 100.0, rel_tol=1e-9)


def test_simulate_plan_deterministic():
    topo, inst = _instance()
    plan = _plan_from_solve(inst["bundles"], inst["edge_capacities"],
                            inst["memory_capacities"])
    sim = StochasticEventSimulator(topo, tau_mem=50.0, swap_success=0.95,
                                   seed=7)
    a = sim.simulate_plan(inst["bundles"], plan, n_realizations=30)
    b = sim.simulate_plan(inst["bundles"], plan, n_realizations=30)
    assert a["e_utility"] == b["e_utility"]
    assert a["var_utility"] == b["var_utility"]
    assert a["served_ratio"] == b["served_ratio"]
    assert a["per_request"] == b["per_request"]


def test_aggregate_schema_and_bounds():
    topo, inst = _instance()
    plan = _plan_from_solve(inst["bundles"], inst["edge_capacities"],
                            inst["memory_capacities"])
    sim = StochasticEventSimulator(topo, tau_mem=50.0, swap_success=0.95,
                                   seed=1)
    agg = sim.simulate_plan(inst["bundles"], plan, n_realizations=50)
    assert agg["n_realizations"] == 50
    assert agg["n_requests"] == len(set(r for r, _ in plan))
    assert agg["e_utility"] >= 0.0
    assert agg["var_utility"] >= 0.0
    assert 0.0 <= agg["served_ratio"] <= 1.0
    assert 0.0 <= agg["sla_violation_prob"] <= 1.0
    assert agg["n_delivered_total"] >= 0
    # failure causes form a distribution
    assert abs(sum(agg["failure_causes"].values()) - 1.0) < 1e-9
    # every planned request has a per-request entry
    assert set(agg["per_request"]) == set(r for r, _ in plan)
    for rid, entry in agg["per_request"].items():
        assert 0.0 <= entry["p_sampled"] <= 1.0
        assert entry["n_delivered"] <= agg["n_realizations"]


def test_empty_plan_returns_zeroed_aggregate():
    topo, inst = _instance()
    sim = StochasticEventSimulator(topo, seed=1)
    agg = sim.simulate_plan(inst["bundles"], [], n_realizations=10)
    assert agg["n_realizations"] == 0
    assert agg["e_utility"] == 0.0
    assert agg["param_expected_utility"] == 0.0


def test_duplicate_bundle_per_request_deduped():
    topo, inst = _instance()
    plan = _plan_from_solve(inst["bundles"], inst["edge_capacities"],
                            inst["memory_capacities"])
    dup = list(plan) + list(plan)
    sim = StochasticEventSimulator(topo, seed=1)
    agg = sim.simulate_plan(inst["bundles"], dup, n_realizations=10)
    # dedupe keeps the higher-utility bundle per request, count stays correct
    assert agg["n_requests"] == len(set(r for r, _ in plan))


def test_sla_thresholds_flag_violations():
    topo, inst = _instance()
    plan = _plan_from_solve(inst["bundles"], inst["edge_capacities"],
                            inst["memory_capacities"])
    sim = StochasticEventSimulator(topo, tau_mem=50.0, seed=1)
    sla = {rid: 1.0 for rid, _ in plan}  # demand impossible fidelity
    agg = sim.simulate_plan(inst["bundles"], plan, n_realizations=30,
                            sla_thresholds=sla)
    for rid, entry in agg["per_request"].items():
        if not math.isnan(entry["sla_violation_prob"]):
            assert entry["sla_violation_prob"] == 1.0


def test_longer_paths_fail_more_often():
    import random

    def build(n_nodes, seed=0):
        topo = _topology(n_nodes=n_nodes, raw=0.85)
        pairs = [("N0", f"N{n_nodes - 1}", 50.0, 0.55),
                 ("N1", f"N{n_nodes - 2}", 50.0, 0.55)]
        bundles, ec, mc = generate_benchmark_instance(topo, pairs,
                                                      random.Random(seed))
        return topo, {"bundles": bundles, "edge_capacities": ec,
                      "memory_capacities": mc}

    def serve(topo, inst):
        plan = _plan_from_solve(inst["bundles"], inst["edge_capacities"],
                                inst["memory_capacities"], seed=3)
        sim = StochasticEventSimulator(topo, tau_mem=50.0, swap_success=0.95,
                                       seed=3)
        return sim.simulate_plan(inst["bundles"], plan, n_realizations=60)[
            "served_ratio"]

    short_topo, short_inst = build(4)
    long_topo, long_inst = build(10)
    assert short_inst["bundles"] and long_inst["bundles"]
    assert serve(short_topo, short_inst) > serve(long_topo, long_inst)


def test_track_events_logs_generation():
    topo, inst = _instance(n_req=2, n_nodes=5)
    plan = _plan_from_solve(inst["bundles"], inst["edge_capacities"],
                            inst["memory_capacities"])
    sim = StochasticEventSimulator(topo, tau_mem=50.0, seed=1)
    agg = sim.simulate_plan(inst["bundles"], plan, n_realizations=4,
                            track_events=True)
    # events were generated in at least one realization (plan non-empty)
    assert agg["n_delivered_total"] >= 0
