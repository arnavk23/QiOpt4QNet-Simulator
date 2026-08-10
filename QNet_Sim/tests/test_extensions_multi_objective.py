"""Tests for the multi-objective allocation module (Extension 9)."""

import random as _random

import pytest

from extensions.multi_objective import (
    MAXIMIZE, MINIMIZE, selection_objectives, pareto_frontier,
    solve_weighted, constraint_frontier, weighted_score,
)
from experiments.instances import (
    generate_grid_topology, generate_benchmark_instance,
)


def _instance(n_req=4, seed=3):
    topo = generate_grid_topology(rows=3, cols=3, edge_capacity=6,
                                  memory_capacity=10)
    rng = _random.Random(seed)
    pairs = []
    for _ in range(n_req):
        src, dst = rng.sample(topo["nodes"], 2)
        pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.4, 0.7)))
    return generate_benchmark_instance(topo, pairs, rng)


def test_selection_objectives_aggregation():
    bundles, ec, mc = _instance(n_req=3)
    sel = [(b["request_id"], b["bundle_id"]) for b in bundles[:3]]
    objs = selection_objectives(bundles, sel)
    assert set(objs) == {"throughput", "fidelity", "success", "latency",
                         "memory", "bell_pairs"}
    assert objs["throughput"] == len(sel)
    assert 0.0 <= objs["fidelity"] <= 1.0
    assert objs["memory"] > 0
    # empty selection
    empty = selection_objectives(bundles, [])
    assert empty["throughput"] == 0 and empty["fidelity"] == 0.0


def test_weighted_score_directions():
    objs = {"throughput": 5, "fidelity": 0.8, "latency": 10, "memory": 20}
    s1 = weighted_score(objs, {"throughput": 1.0})
    s2 = weighted_score(objs, {"latency": 1.0})
    assert s1 == pytest.approx(5.0)
    assert s2 == pytest.approx(-10.0)
    with pytest.raises(ValueError):
        weighted_score(objs, {"bogus": 1.0})


def test_pareto_frontier_is_non_dominated():
    bundles, ec, mc = _instance(n_req=3)
    front = pareto_frontier(bundles, ec, mc, max_combos=20000)
    assert front, "frontier should be non-empty"
    pts = [p["objectives"] for p in front]
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i == j:
                continue
            # a must not dominate b and b must not dominate a
            assert not _dominates(a, b), f"frontier point {i} dominates {j}"
            assert not _dominates(b, a), f"frontier point {j} dominates {i}"


def _dominates(a, b):
    at_least = all(a[k] >= b[k] - 1e-12 for k in MAXIMIZE)
    at_least &= all(a[k] <= b[k] + 1e-12 for k in MINIMIZE)
    strictly = any(a[k] > b[k] + 1e-12 for k in MAXIMIZE)
    strictly |= any(a[k] < b[k] - 1e-12 for k in MINIMIZE)
    return at_least and strictly


def test_pareto_frontier_dominates_all_enumerated_points():
    """Every enumerated feasible selection must be dominated by some
    frontier point (or be on the frontier itself)."""
    from extensions.multi_objective import _selections_iter, _feasible
    bundles, ec, mc = _instance(n_req=3)
    front = pareto_frontier(bundles, ec, mc, max_combos=20000)
    front_obj = [p["objectives"] for p in front]
    for selections in _selections_iter(bundles, max_combos=20000):
        if not _feasible(selections, bundles, ec, mc):
            continue
        sel_list = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        objs = selection_objectives(bundles, sel_list)
        dominated = any(_dominates(fo, objs) for fo in front_obj)
        assert dominated or any(objs == fo for fo in front_obj)


def test_solve_weighted_respects_weights():
    bundles, ec, mc = _instance(n_req=3)
    best_throughput = solve_weighted(bundles, ec, mc,
                                     {"throughput": 1.0}, max_combos=20000)
    n_tp = sum(1 for v in best_throughput.values() if v is not None)
    # throughput-first should serve as many requests as feasible
    best_latency = solve_weighted(bundles, ec, mc,
                                  {"latency": 1.0}, max_combos=20000)
    lat_tp = sum(1 for v in best_latency.values() if v is not None)
    assert n_tp >= lat_tp


def test_constraint_frontier_feasibility():
    bundles, ec, mc = _instance(n_req=3)
    rows = constraint_frontier(bundles, ec, mc, targets=[0.4, 0.5, 0.6],
                               constrain="fidelity", maximize="throughput",
                               max_combos=20000)
    assert rows
    feasible = [r for r in rows if r["feasible"]]
    assert feasible
    for r in feasible:
        assert r["fidelity"] + 1e-12 >= r["target"]
    # higher fidelity targets never serve more requests
    tp = [r["throughput"] for r in feasible]
    assert tp == sorted(tp, reverse=True)
