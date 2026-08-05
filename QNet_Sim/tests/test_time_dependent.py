import math

import pytest

from optimization.time_dependent_optimizer import (
    _poisson, _poisson_trace, fidelity_after_hold,
    run_time_dependent_comparison,
)
from optimization.streaming_annealer import StreamingAnnealer


def test_poisson_distribution():
    rng = __import__("random").Random(0)
    counts = [_poisson(rng, 5.0) for _ in range(2000)]
    assert 3.5 < sum(counts) / len(counts) < 6.5


def test_poisson_zero_rate():
    rng = __import__("random").Random(1)
    assert _poisson(rng, 0.0) == 0
    assert _poisson(rng, -1.0) == 0


def test_poisson_trace_shape():
    topo = _chain_topo()
    trace = _poisson_trace(topo, n_slots=5, mean_rate=2.0, seed=0)
    assert len(trace) == 5
    total_bundles = 0
    for slot in trace:
        for rid, bundles in slot.items():
            assert rid
            for b in bundles:
                total_bundles += 1
                assert b["request_id"] == rid
                assert "latency" in b and "fidelity" in b and "utility" in b
    assert total_bundles > 0


def test_fidelity_after_hold_decays():
    f0 = 0.9
    f_short = fidelity_after_hold(f0, hold_time=0.001, tau_mem=1000.0)
    f_long = fidelity_after_hold(f0, hold_time=100.0, tau_mem=1.0)
    assert f_short == pytest.approx(f0, rel=1e-3)
    assert f_long < f0 * 0.5


def test_fidelity_after_hold_zero_hold_identity():
    assert fidelity_after_hold(0.9, 0.0, 5.0) == pytest.approx(0.9)
    assert fidelity_after_hold(0.9, 1.0, 0.0) == pytest.approx(0.9)


def test_fidelity_aware_risk_energy():
    ec, mc = _caps()
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}, lat=10.0, fid=0.9),
               _b("b1", "r1", 50.0, {("A", "R"): 1}, {"A": 1}, lat=80.0, fid=0.95)]
    sa = StreamingAnnealer(ec, mc, seed=0, risk_weight=2.0, risk_tau=40.0,
                           use_fidelity_risk=True, hold_scale=6.0)
    sa.add_request("r1", bundles)
    e_low = sa._energy({"r1": "b0"})
    e_high = sa._energy({"r1": "b1"})
    # the high-latency bundle's delivered fidelity decays more, so its risk
    # term is larger -> higher energy.
    assert e_high > e_low


def test_time_dependent_comparison_runs():
    topo = _chain_topo()
    res = run_time_dependent_comparison(topo, n_slots=3, mean_rate=1.0,
                                        tau_mem=5.0, seed=3)
    for name in ("decoherence_aware", "static"):
        r = res[name]
        assert "served" in r and "utility" in r and "mean_delivered_fidelity" in r
        assert r["utility"] >= 0.0


def _b(bid, rid, util, edge_d, mem_d, lat=0.0, fid=0.9):
    return {
        "bundle_id": bid, "request_id": rid, "utility": util,
        "edge_demands": edge_d, "memory_demands": mem_d,
        "latency": lat, "fidelity": fid, "path": ["A", "R", "B"],
    }


def _caps():
    return {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}


def _chain_topo():
    from experiments.instances import generate_chain_topology
    return lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
