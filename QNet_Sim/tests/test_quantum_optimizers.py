from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer


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
