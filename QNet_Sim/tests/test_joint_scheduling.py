"""Tests for the temporal memory model, dynamic traffic and the joint
routing + memory-scheduling optimizer (Extensions 1, 3, 4, 18)."""

import math

import pytest

from routing.memory_scheduler import (
    EntangledPair, EntanglementAgeTracker, MemoryScheduler,
    temporal_memory_windows,
)
from routing.temporal_request import (
    TemporalRequest, generate_dynamic_trace, batch_trace_by_slot,
)
from optimization.joint_scheduler import (
    JointRoutingScheduler, delivered_fidelity, generate_temporal_bundles,
)
from optimization.online_optimizers import (
    static_solve, online_solve, receding_horizon_solve, run_regime_comparison,
)
from experiments.instances import generate_chain_topology


# ----------------------------------------------------------------------
# temporal memory model
# ----------------------------------------------------------------------
def test_entangled_pair_fidelity_decays_and_is_monotone():
    p = EntangledPair(pair_id="p1", node="A", generation_time=1.0,
                      initial_fidelity=0.9, t2=5.0)
    assert p.fidelity_at(1.0) == pytest.approx(0.9)
    f_early = p.fidelity_at(3.0)
    f_late = p.fidelity_at(20.0)
    assert f_early > f_late
    assert p.fidelity_at(1e6) < 0.3  # decays towards the mixed-state floor


def test_entangled_pair_expiry():
    p = EntangledPair(pair_id="p1", node="A", generation_time=0.0,
                      initial_fidelity=0.9, t2=5.0)
    assert not p.is_expired(4.9, tau_mem=5.0)
    assert p.is_expired(5.0, tau_mem=5.0)


def test_age_tracker_decisions():
    tr = EntanglementAgeTracker(discard_floor=0.5, purify_threshold=0.9,
                                tau_mem=10.0)
    fresh = EntangledPair(pair_id="fresh", node="A", generation_time=0.0,
                          initial_fidelity=0.99, t2=50.0)
    aged = EntangledPair(pair_id="aged", node="A", generation_time=0.0,
                         initial_fidelity=0.6, t2=5.0)
    tr.add(fresh)
    tr.add(aged)
    assert tr.decide("fresh", 0.5) == "keep"
    assert tr.decide("aged", 0.5) in ("purify", "discard")
    assert tr.decide("missing", 0.0) == "replace"
    tr.remove("aged")
    assert "aged" not in tr.pairs


def test_memory_scheduler_rejects_overflow():
    sched = MemoryScheduler({"A": 2})
    assert sched.reserve("A", 2, 0.0, 5.0) is not None
    # second overlapping reservation exceeds capacity at every instant
    assert sched.reserve("A", 1, 1.0, 4.0) is None
    # disjoint in time is fine
    assert sched.reserve("A", 2, 5.0, 8.0) is not None


def test_memory_scheduler_occupancy_and_peak():
    sched = MemoryScheduler({"A": 4})
    r1 = sched.reserve("A", 2, 1.0, 4.0)
    r2 = sched.reserve("A", 1, 2.0, 6.0)
    assert r1 is not None and r2 is not None
    assert sched.occupancy("A", 0.0) == 0
    assert sched.occupancy("A", 2.0) == 3
    assert sched.occupancy("A", 5.0) == 1
    assert sched.peak_occupancy("A") == 3
    sched.release(r1)
    assert sched.occupancy("A", 2.0) == 1


def test_memory_scheduler_utilization_bounded():
    sched = MemoryScheduler({"A": 2})
    sched.reserve("A", 1, 0.0, 10.0)
    sched.reserve("A", 1, 0.0, 10.0)
    assert 0.0 <= sched.utilization(0.0, 10.0, 1.0) <= 1.0


def test_temporal_memory_windows():
    bundle = {"memory_demands": {"A": 2, "B": 1}, "latency": 5.0}
    windows = temporal_memory_windows(bundle, arrival_time=3.0, tau_mem=5.0)
    assert windows["A"] == (3.0, 8.0)
    assert windows["B"] == (3.0, 8.0)


# ----------------------------------------------------------------------
# dynamic traffic
# ----------------------------------------------------------------------
def test_temporal_request_interface():
    topo = generate_chain_topology(n_nodes=4)
    req = TemporalRequest(request_id="r0", source="N0", destination="N3",
                          minimum_fidelity=0.6, weight=20.0, arrival=1.0,
                          deadline=4.0, priority=0.8, n_pairs=2)
    assert req.completion_ok(3.9)
    assert not req.completion_ok(4.1)
    assert req.slack(3.0) == pytest.approx(1.0)
    legacy = req.to_request()
    assert legacy.source == "N0" and legacy.destination == "N3"


def test_generate_dynamic_trace_deterministic_and_sorted():
    topo = generate_chain_topology(n_nodes=6)
    a = generate_dynamic_trace(topo, n_slots=8, mean_rate=1.2, seed=42)
    b = generate_dynamic_trace(topo, n_slots=8, mean_rate=1.2, seed=42)
    assert [r.request_id for r in a] == [r.request_id for r in b]
    arrivals = [r.arrival for r in a]
    assert arrivals == sorted(arrivals)
    for r in a:
        assert r.max_latency <= r.deadline - r.arrival + 1e-9


def test_batch_trace_by_slot():
    topo = generate_chain_topology(n_nodes=4)
    trace = generate_dynamic_trace(topo, n_slots=5, mean_rate=1.5, seed=7)
    batches = batch_trace_by_slot(trace, slot_duration=1.0)
    flat = [r for batch in batches for r in batch]
    assert [r.request_id for r in flat] == [r.request_id for r in trace]


# ----------------------------------------------------------------------
# joint scheduler
# ----------------------------------------------------------------------
def test_delivered_fidelity_decays_with_hold():
    bundle = {"fidelity": 0.9}
    assert delivered_fidelity(bundle, hold_time=0.0, tau_mem=5.0) == pytest.approx(0.9)
    assert delivered_fidelity(bundle, 1.0, 5.0) > delivered_fidelity(bundle, 20.0, 5.0)


def _small_trace(n_slots=4, mean_rate=0.8, seed=42):
    topo = generate_chain_topology(n_nodes=6, edge_capacity=4,
                                   memory_capacity=6)
    trace = generate_dynamic_trace(topo, n_slots=n_slots, mean_rate=mean_rate,
                                   deadline_horizon=4.0, seed=seed)
    bundles = generate_temporal_bundles(topo, trace)
    return topo, trace, bundles


def test_generate_temporal_bundles_tags():
    topo, trace, bundles = _small_trace()
    assert set(bundles) == {r.request_id for r in trace}
    for rid, bs in bundles.items():
        assert bs
        for b in bs:
            assert b["request_id"] == rid
            assert b["arrival"] >= 0.0
            assert "deadline" in b and "priority" in b and "n_pairs" in b
            assert b["t_consume"] >= b["t_gen"]
            assert b["hold_time"] == pytest.approx(b["t_consume"] - b["t_gen"])


def test_joint_scheduler_deterministic():
    topo, trace, bundles = _small_trace()
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    s1 = JointRoutingScheduler(ec, mc, tau_mem=5.0, seed=1)
    for rid, bs in bundles.items():
        s1.add_request(rid, bs)
    r1 = s1.solve(max_iterations=200, n_restarts=1)
    s2 = JointRoutingScheduler(ec, mc, tau_mem=5.0, seed=1)
    for rid, bs in bundles.items():
        s2.add_request(rid, bs)
    r2 = s2.solve(max_iterations=200, n_restarts=1)
    assert r1["decisions"] == r2["decisions"]
    assert r1["served"] == r2["served"]


def test_joint_schedule_meets_hard_deadlines():
    topo, trace, bundles = _small_trace()
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    sched = JointRoutingScheduler(ec, mc, tau_mem=5.0, hard_deadline=True, seed=3)
    for rid, bs in bundles.items():
        sched.add_request(rid, bs)
    res = sched.solve(max_iterations=300, n_restarts=1)
    for d in res["decisions"]:
        assert d["completion_time"] <= d["deadline"] + 1e-9
    assert 0.0 <= res["served_ratio"] <= 1.0
    assert 0.0 <= res["mean_delivered_fidelity"] <= 1.0
    assert 0.0 <= res["mean_memory_utilization"] <= 1.0
    # no expired bundles (hold <= tau_mem)
    for d in res["decisions"]:
        assert d["hold_time"] <= 5.0 + 1e-9


def test_joint_schedule_memory_feasible_at_all_times():
    """The whole point of the joint scheduler: after the solve, occupancy on
    every node stays within capacity at every sampled instant."""
    topo, trace, bundles = _small_trace(mean_rate=1.2, seed=11)
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    sched = JointRoutingScheduler(ec, mc, tau_mem=5.0, seed=4)
    for rid, bs in bundles.items():
        sched.add_request(rid, bs)
    res = sched.solve(max_iterations=300, n_restarts=1)

    occ = {n: [0] * 60 for n in mc}
    for d in res["decisions"]:
        b = None
        for bb in bundles[d["request_id"]]:
            if bb["bundle_id"] == d["bundle_id"]:
                b = bb
                break
        assert b is not None
        start = b["arrival"]
        for t_idx in range(60):
            t = t_idx / 10.0
            if start <= t < start + b["latency"]:
                for n, demand in b["memory_demands"].items():
                    occ[n][t_idx] += demand
    for n, series in occ.items():
        assert max(series) <= mc[n], f"node {n} over capacity"


# ----------------------------------------------------------------------
# online / receding-horizon regimes
# ----------------------------------------------------------------------
def test_control_regime_metrics_shape():
    topo, trace, bundles = _small_trace(n_slots=3, mean_rate=0.7, seed=5)
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    kw = dict(max_iterations=150, n_restarts=1)
    for fn, args in [(static_solve, (ec, mc, bundles)),
                     (online_solve, (ec, mc, trace, bundles)),
                     (receding_horizon_solve, (ec, mc, trace, bundles))]:
        m = fn(*args, tau_mem=5.0, seed=5, **kw)
        assert m["regime"] in ("static", "online", "receding_horizon")
        assert set(m) >= {"served", "served_ratio", "on_time_ratio",
                          "mean_delivered_fidelity", "weighted_on_time",
                          "wall_time_s"}
        assert 0.0 <= m["served_ratio"] <= 1.0


def test_run_regime_comparison_runs():
    topo_fn = lambda: generate_chain_topology(n_nodes=5, edge_capacity=4,
                                              memory_capacity=6)
    res = run_regime_comparison(topo_fn, n_slots=3, mean_rate=0.6, tau_mem=5.0,
                                window_size=2, seed=9)
    assert res["n_requests"] >= 0
    assert {r["regime"] for r in res["rows"]} == {"static", "online",
                                                  "receding_horizon"}
