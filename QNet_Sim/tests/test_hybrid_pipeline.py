"""Tests for the four-stage hybrid pipeline (Extension 6)."""

import pytest

from optimization.hybrid_pipeline import (
    CandidateReducer, HybridPipeline, _dominance_prune, run_hybrid_comparison,
)
from experiments.instances import (
    generate_chain_topology, generate_benchmark_instance,
)


def _instance(n_req=5, seed=1):
    import random as _random
    topo = generate_chain_topology(n_nodes=6, edge_capacity=4,
                                   memory_capacity=6)
    rng = _random.Random(seed)
    pairs = []
    for _ in range(n_req):
        src, dst = rng.sample(topo["nodes"], 2)
        pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))
    return generate_benchmark_instance(topo, pairs, rng)


def test_reducer_filters_and_prunes():
    bundles, ec, mc = _instance()
    red = CandidateReducer(fidelity_threshold=0.6, max_latency=100.0)
    out = red.reduce(bundles)
    for b in out:
        assert b.get("fidelity", 0.0) >= 0.6
    assert len(out) <= len(bundles)


def test_dominance_prune_keeps_undominated():
    bundles, _, _ = _instance()
    pruned = _dominance_prune(bundles)
    # no remaining bundle is strictly dominated on (fidelity, cost, latency)
    for i, bc in enumerate(pruned):
        for j, bo in enumerate(pruned):
            if i == j:
                continue
            assert not (
                bo.get("fidelity", 0.0) >= bc.get("fidelity", 0.0)
                and bo.get("bell_pair_cost", float("inf")) <= bc.get("bell_pair_cost", float("inf"))
                and bo.get("latency", float("inf")) <= bc.get("latency", float("inf"))
                and (bo.get("fidelity", 0.0) > bc.get("fidelity", 0.0)
                     or bo.get("bell_pair_cost", float("inf")) < bc.get("bell_pair_cost", float("inf"))
                     or bo.get("latency", float("inf")) < bc.get("latency", float("inf")))
            ), f"{bc['bundle_id']} dominated by {bo['bundle_id']}"


def test_hybrid_pipeline_returns_feasible_selection():
    bundles, ec, mc = _instance(n_req=4)
    pipe = HybridPipeline(bundles, ec, mc, seed=42)
    res = pipe.solve(keep_per_request=4, num_reads=10, refine_iterations=300)
    sel = res["selected"]
    assert pipe.stats["violations"] == 0
    assert pipe.stats["served"] == len(set(rid for rid, _ in sel))
    # feasibility: no edge or memory demand exceeds capacity
    edge_load = {}
    mem_load = {}
    by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    for rid, bid in sel:
        b = by_key[(rid, bid)]
        for e, d in b["edge_demands"].items():
            edge_load[tuple(sorted(e))] = edge_load.get(tuple(sorted(e)), 0) + d
        for n, d in b["memory_demands"].items():
            mem_load[n] = mem_load.get(n, 0) + d
    for e, load in edge_load.items():
        assert load <= ec[e]
    for n, load in mem_load.items():
        assert load <= mc[n]


def test_hybrid_pipeline_reduces_candidate_count():
    bundles, ec, mc = _instance(n_req=5)
    pipe = HybridPipeline(bundles, ec, mc, seed=42)
    reduced = pipe.stage1_reduce(keep_per_request=3)
    assert len(reduced) <= len(bundles)
    assert pipe.stats["n_candidates_in"] == len(bundles)
    assert pipe.stats["n_candidates_out"] == len(reduced)


def test_run_hybrid_comparison_stages():
    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=4,
                                              memory_capacity=6)
    res = run_hybrid_comparison(topo_fn, n_requests=4, seed=42)
    assert res["n_bundles_reduced"] <= res["n_bundles_in"]
    for stage in ("full_pipeline", "qubo_only", "qubo_plus_repair"):
        assert res[stage]["utility"] >= 0.0
        assert res[stage]["served"] >= 0
