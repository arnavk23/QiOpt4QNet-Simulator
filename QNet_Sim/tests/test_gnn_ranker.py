"""Tests for the GraphSAGE-style GNN candidate ranker (Extension 8)."""

import numpy as np
import pytest

from baselines.gnn_ranker import (
    GraphFeatureBuilder, GraphSAGERanker, gnn_guided_topk, gnn_guided_qubo,
)
from experiments.instances import (
    generate_chain_topology, generate_benchmark_instance,
)


def _topo():
    return generate_chain_topology(n_nodes=6, edge_capacity=4,
                                   memory_capacity=6)


def _bundles(n_req=4, seed=2):
    import random as _random
    topo = _topo()
    rng = _random.Random(seed)
    pairs = []
    for _ in range(n_req):
        src, dst = rng.sample(topo["nodes"], 2)
        pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))
    return topo, generate_benchmark_instance(topo, pairs, rng)


def test_graph_feature_builder_shapes():
    topo = _topo()
    g = GraphFeatureBuilder(topo)
    assert g.node_feats.shape == (len(topo["nodes"]), 4)
    assert g.edge_feat_dim() == 4
    n0 = topo["nodes"][0]
    assert g.neighbors(n0), "first node must have a neighbour on a chain"
    # path pooling works
    path = list(topo["nodes"])
    assert len(g.path_node_indices(path)) == len(topo["nodes"])
    assert g.path_edge_mean(path).shape == (4,)


def test_ranker_scores_are_float_and_deterministic():
    topo = _topo()
    ranker = GraphSAGERanker(topo, seed=5)
    path = [topo["nodes"][0], topo["nodes"][1], topo["nodes"][2]]
    s1 = ranker.score(path)
    s2 = ranker.score(path)
    assert isinstance(s1, float)
    assert s1 == s2  # no RNG inside forward pass


def test_fit_reduces_loss():
    topo, (bundles, ec, mc) = _bundles(n_req=3)
    ranker = GraphSAGERanker(topo, seed=5, lr=0.05)
    targets = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    before = ranker.fit(bundles, targets, epochs=1)["final_loss"]
    # after more training the MSE should drop
    after = ranker.fit(bundles, targets, epochs=40)["final_loss"]
    assert after < before


def test_fit_requires_targets():
    topo = _topo()
    ranker = GraphSAGERanker(topo, seed=5)
    with pytest.raises(ValueError):
        ranker.fit([], {})


def test_gnn_guided_topk_returns_topk_per_request():
    topo, (bundles, ec, mc) = _bundles(n_req=3)
    ranker, reduced, loss = gnn_guided_topk(topo, bundles, k=2,
                                            train_fraction=1.0, epochs=5,
                                            seed=42)
    counts = {}
    for b in reduced:
        counts[b["request_id"]] = counts.get(b["request_id"], 0) + 1
    assert max(counts.values()) <= 2
    assert isinstance(loss, float)


def test_gnn_guided_qubo_solves():
    topo, (bundles, ec, mc) = _bundles(n_req=3)
    r = gnn_guided_qubo(topo, bundles, ec, mc, k=3, num_reads=10, seed=42)
    assert r["served"] >= 0
    assert r["utility"] >= 0.0
    assert r["n_bundles_reduced"] <= len(bundles)
    # feasibility check on the returned selection
    by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    edge_load = {}
    mem_load = {}
    for rid, bid in r["selected"]:
        b = by_key[(rid, bid)]
        for e, d in b["edge_demands"].items():
            edge_load[tuple(sorted(e))] = edge_load.get(tuple(sorted(e)), 0) + d
        for n, d in b["memory_demands"].items():
            mem_load[n] = mem_load.get(n, 0) + d
    for e, load in edge_load.items():
        assert load <= ec[e]
    for n, load in mem_load.items():
        assert load <= mc[n]
