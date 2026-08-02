from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
from optimization.sequential_branch_optimizer import SequentialBranchOptimizer


def _make_bundle(bundle_id, request_id, utility, edge_demands, memory_demands, path=None):
    return {
        "bundle_id": bundle_id,
        "request_id": request_id,
        "path": path or ["A", "R", "B"],
        "edge_demands": edge_demands,
        "memory_demands": memory_demands,
        "utility": utility,
    }


def test_metropolis_single_request_selects_best():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
        _make_bundle("b2", "r1", 10.0, {("A", "R"): 4, ("R", "B"): 4}, {"A": 4, "R": 8, "B": 4}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=1000)
    selected = {rid for rid, bid in result["selected"]}
    assert "r1" in selected
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b0"


def test_metropolis_two_requests():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=1000)
    assert len(result["selected"]) == 2


def test_metropolis_respects_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=2000, penalty=100.0, edge_penalty=50.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_metropolis_greedy_seed():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    seed = opt._greedy_seed()
    assert seed["r1"] == "b1"


def test_tensor_network_single_request():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b0"


def test_tensor_network_two_requests():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r2", 10.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    assert len(result["selected"]) == 2


def test_tensor_network_respects_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(edge_penalty=50.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_feasibility_repair():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    selections = {"r1": "b0"}
    repaired = opt._feasibility_repair(selections)
    assert repaired["r1"] == "b1"

    selections2 = {"r1": "b1"}
    repaired2 = opt._feasibility_repair(selections2)
    assert repaired2["r1"] == "b1"

    selections3 = {"r1": None}
    repaired3 = opt._feasibility_repair(selections3)
    assert repaired3["r1"] is None


def test_validates_missing_fields():
    import pytest
    with pytest.raises(ValueError, match="missing"):
        MetropolisAnnealer([{"bundle_id": "b0"}], {}, {})


def test_tensor_network_validates_missing_fields():
    import pytest
    with pytest.raises(ValueError, match="missing"):
        TensorNetworkOptimizer([{"bundle_id": "b0"}], {}, {})


def _congested_bundles():
    # Two requests whose high-utility bundles concentrate load on a shared
    # edge (E,R); low-utility alternatives use less capacity.
    return [
        _make_bundle("b_hi", "r1", 60.0, {("E", "R"): 2}, {"M": 1}),
        _make_bundle("b_lo", "r1", 30.0, {("E", "R"): 1}, {"M": 1}),
        _make_bundle("b_hi", "r2", 50.0, {("E", "R"): 2}, {"M": 1}),
        _make_bundle("b_lo", "r2", 20.0, {("E", "R"): 1}, {"M": 1}),
    ]


def _load_from_selections(opt, selections):
    edge_load = {e: 0 for e in opt.edge_capacities}
    mem_load = {n: 0 for n in opt.memory_capacities}
    for b in opt.bundles:
        if selections.get(b["request_id"]) == b["bundle_id"]:
            for e, d in b["edge_demands"].items():
                edge_load[tuple(sorted(e))] = edge_load.get(tuple(sorted(e)), 0) + d
            for n, d in b["memory_demands"].items():
                mem_load[n] = mem_load.get(n, 0) + d
    return edge_load, mem_load


def test_metropolis_congestion_penalty_discourages_concentration():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=1)
    opt._B, opt._C = 100.0, 10.0
    opt._D, opt._E = 100.0, 0.0

    concentrated = {"r1": "b_hi", "r2": "b_hi"}   # load 4 == cap; congestion C*16
    spread = {"r1": "b_hi", "r2": "b_lo"}          # load 3 < cap; congestion C*9

    # With the congestion term the quadratic penalty flips the ordering
    assert opt._energy(spread) < opt._energy(concentrated)
    # without it the higher-utility concentrated choice wins.
    opt._C = 0.0
    assert opt._energy(concentrated) < opt._energy(spread)


def test_metropolis_delta_energy_matches_full_energy():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=7)
    selections = opt._greedy_seed()
    load, mem_load = _load_from_selections(opt, selections)

    rid = "r2"
    prev = selections[rid]
    new = "b_lo" if prev != "b_lo" else "b_hi"
    delta = opt._energy_delta(rid, prev, new, load, mem_load)

    old_state = dict(selections)
    new_state = dict(selections)
    new_state[rid] = new
    full_delta = opt._energy(new_state) - opt._energy(old_state)

    assert abs(delta - full_delta) < 1e-9


def test_metropolis_delta_apply_consistent():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=7)
    selections = opt._greedy_seed()
    load, mem_load = _load_from_selections(opt, selections)

    rid = "r2"
    prev = selections[rid]
    new = "b_lo" if prev != "b_lo" else "b_hi"
    opt._apply_delta(rid, prev, new, load, mem_load)
    selections[rid] = new

    expect_load, expect_mem = _load_from_selections(opt, selections)
    assert load == expect_load
    assert mem_load == expect_mem


def test_metropolis_restarts_find_feasible():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    result = opt.solve(max_iterations=500, penalty=100.0, edge_penalty=50.0,
                       memory_penalty=50.0, n_restarts=5)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_branch_expander_single_request():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
        _make_bundle("b2", "r1", 10.0, {("A", "R"): 4, ("R", "B"): 4}, {"A": 4, "R": 8, "B": 4}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b0"


def test_branch_expander_two_requests():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r2", 10.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
    ]
    edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    mem_caps = {"A": 10, "R": 10, "B": 10}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve()
    assert len(result["selected"]) == 2


def test_branch_expander_respects_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
    ]
    edge_caps = {("A", "R"): 5, ("R", "B"): 5}
    mem_caps = {"A": 5, "R": 5, "B": 5}

    opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
    result = opt.solve(edge_penalty=50.0, memory_penalty=50.0)
    result_bid = {rid: bid for rid, bid in result["selected"]}
    assert result_bid["r1"] == "b1"


def test_tensor_network_greedy_seed_respects_combined_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 2}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
    seed = opt._greedy_seed()
    assert seed["r1"] == "b0"
    assert seed["r2"] is None


def test_metropolis_greedy_seed_respects_combined_capacity():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 2}, {"M": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 2}, {"M": 1}),
    ]
    edge_caps = {("A", "R"): 3}
    mem_caps = {"M": 100}

    opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=42)
    seed = opt._greedy_seed()
    assert seed["r1"] == "b0"
    assert seed["r2"] is None


def test_branch_expander_deterministic():
    bundles = _congested_bundles()
    edge_caps = {("E", "R"): 4}
    mem_caps = {"M": 100}

    r1 = SequentialBranchOptimizer(bundles, edge_caps, mem_caps).solve()
    r2 = SequentialBranchOptimizer(bundles, edge_caps, mem_caps).solve()
    assert r1["selected"] == r2["selected"]
