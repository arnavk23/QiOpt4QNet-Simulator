"""Tests for the adaptive candidate-reduction QUBO (Extension 7)."""

import pytest

from optimization.adaptive_qubo import (
    AdaptiveCandidateSelector, adaptive_qubo_solve, reference_solution,
    run_topk_sweep,
)
from experiments.instances import (
    generate_chain_topology, generate_benchmark_instance,
    contention_sweep_instances,
)


def _instance(n_req=6, seed=42):
    topo = generate_chain_topology(n_nodes=6, edge_capacity=4,
                                   memory_capacity=6)
    return contention_sweep_instances(lambda: topo, [n_req], seed=seed)[
        f"req{n_req}"]


def _flat_bundles():
    import random as _random
    topo = generate_chain_topology(n_nodes=6, edge_capacity=4,
                                   memory_capacity=6)
    rng = _random.Random(1)
    pairs = []
    for _ in range(4):
        src, dst = rng.sample(topo["nodes"], 2)
        pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))
    return generate_benchmark_instance(topo, pairs, rng)


def test_selector_reduces_per_request_topk():
    bundles, ec, mc = _flat_bundles()
    sel = AdaptiveCandidateSelector(ec, mc, seed=42)
    reduced = sel.select(bundles, k=2)
    counts = {}
    for b in reduced:
        counts[b["request_id"]] = counts.get(b["request_id"], 0) + 1
    assert max(counts.values()) <= 2
    assert len(reduced) <= len(bundles)


def test_selector_respects_filters():
    bundles, ec, mc = _flat_bundles()
    sel = AdaptiveCandidateSelector(ec, mc, fidelity_threshold=0.9, seed=1)
    reduced = sel.select(bundles, k=8)
    for b in reduced:
        assert b.get("fidelity", 0.0) >= 0.9
    # per-request infeasible bundles dropped
    sel2 = AdaptiveCandidateSelector(ec, mc, seed=2)
    reduced2 = sel2.select(bundles, k=1)
    for b in reduced2:
        for n, d in b["memory_demands"].items():
            assert d <= mc[n]


def test_adaptive_qubo_solve_reduces_candidates():
    inst = _instance(n_req=6)
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    r = adaptive_qubo_solve(b, ec, mc, k=3, num_reads=10, seed=42)
    assert r["n_bundles_in"] == len(b)
    assert r["n_bundles_in_qubo"] <= len(b)
    assert r["n_qubo_variables"] >= 0
    assert 0 <= r["served"] <= inst["n_requests"]
    assert r["utility"] >= 0.0
    # every served request has exactly one selected bundle
    rids = [rid for rid, _ in r["selected"]]
    assert len(rids) == len(set(rids))


def test_reference_uses_full_candidate_set():
    inst = _instance(n_req=6)
    b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
    ref = reference_solution(b, ec, mc, num_reads=10, seed=42)
    assert ref["n_bundles_in_qubo"] == len(b)
    assert ref["k"] is None


def test_topk_sweep_rows_monotone_in_qubo_size():
    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=4,
                                              memory_capacity=6)
    res = run_topk_sweep(topo_fn, n_requests=6, k_values=[2, 8, 32],
                         num_reads=8, seed=42)
    assert res["n_requests"] == 6
    rows = res["rows"]
    assert [r["k"] for r in rows] == [2, 8, 32]
    for k, r in zip([2, 8, 32], rows):
        assert r["n_bundles_in_qubo"] <= res["n_bundles"]
        assert r["relative_gap"] >= 0.0
    # larger k never shrinks the QUBO candidate pool
    qubo_sizes = [r["n_bundles_in_qubo"] for r in rows]
    assert qubo_sizes == sorted(qubo_sizes)
