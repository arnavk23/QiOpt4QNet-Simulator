"""Tests for k-disjoint entanglement path provisioning (Extension 11)."""

import random as _random

import pytest

from extensions.disjoint_paths import (
    k_disjoint_paths, generate_multipath_bundles, _combine_multipath,
    _evaluate_single, run_disjoint_comparison,
)
from experiments.instances import generate_grid_topology


def _grid(rows=3, cols=4):
    return generate_grid_topology(rows=rows, cols=cols, edge_capacity=4,
                                  memory_capacity=12)


def test_k_disjoint_paths_are_edge_disjoint():
    topo = _grid()
    paths = k_disjoint_paths(topo, "G0_0", "G2_3", k=3)
    assert len(paths) >= 2
    for a in paths:
        for b in paths:
            if a is b:
                continue
            edges_a = set(zip(a, a[1:]))
            edges_b = set(zip(b, b[1:]))
            assert edges_a.isdisjoint(edges_b)


def test_generate_multipath_bundles_contains_composites():
    topo = _grid()
    bundles = generate_multipath_bundles(topo, "G0_0", "G2_3",
                                         weight=50.0, min_fidelity=0.5,
                                         k_paths=3)
    multis = [b for b in bundles if b.get("n_paths", 1) > 1]
    assert multis, "grid should admit composite multipath bundles"
    for b in multis:
        assert b["n_paths"] >= 2
        assert b["success_probability"] > 0.0


def test_combine_multipath_improves_success():
    topo = _grid()
    path = ["G0_0", "G1_0", "G2_0"]
    single = _evaluate_single(topo, path, q=0, source="G0_0", target="G2_0",
                              weight=50.0, min_fidelity=0.5)
    assert single is not None
    combined = _combine_multipath([single, single], "G0_0", "G2_0", 50.0,
                                  0.5, bundle_id="m")
    # redundancy: success at least as high, fidelity max, latency max
    assert combined["success_probability"] >= single["success_probability"]
    assert combined["fidelity"] == single["fidelity"]
    assert combined["latency"] >= single["latency"]
    assert combined["n_paths"] == 2


def test_run_disjoint_comparison_multipath_helps():
    topo = _grid()
    res = run_disjoint_comparison(topo, n_requests=4, n_expected_disjoint=2,
                                  seed=42)
    assert res["n_requests"] == 4
    by_set = {r["candidate_set"]: r for r in res["rows"]}
    assert set(by_set) == {"single_path", "multipath"}
    assert res["n_bundles_single"] > 0 and res["n_bundles_multipath"] > 0
    assert 0 <= res["n_multipath_selected"] <= res["n_requests"]
    for r in res["rows"]:
        assert 0 <= r["served"] <= res["n_requests"]
    # multipath provisioning never reduces end-to-end coverage
    assert by_set["multipath"]["served"] >= by_set["single_path"]["served"]
