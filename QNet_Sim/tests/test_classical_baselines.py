from baselines.classical_baselines import (
    ALL_BASELINES,
    CPSAT_AVAILABLE,
    CongestionAwareGreedyAllocator,
    GreedyLocalSearchAllocator,
    HighestFidelityFirstAllocator,
    RandomFeasibleAllocator,
    UtilityPerResourceGreedyAllocator,
    run_all_baselines,
)
from baselines.feasibility import FeasibilityChecker, compute_metrics


def _make_bundle(bundle_id, request_id, utility, edge_demands, memory_demands,
                  path=None, fidelity=None, success_probability=None, bell_pair_cost=None):
    b = {
        "bundle_id": bundle_id,
        "request_id": request_id,
        "path": path or ["A", "R", "B"],
        "edge_demands": edge_demands,
        "memory_demands": memory_demands,
        "utility": utility,
    }
    if fidelity is not None:
        b["fidelity"] = fidelity
    if success_probability is not None:
        b["success_probability"] = success_probability
    if bell_pair_cost is not None:
        b["bell_pair_cost"] = bell_pair_cost
    return b


EDGE_CAPS = {("A", "R"): 10, ("R", "B"): 10}
MEM_CAPS = {"A": 10, "R": 10, "B": 10}


def test_utility_greedy_picks_best_single_request():
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _make_bundle("b1", "r1", 30.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
    ]
    result = UtilityPerResourceGreedyAllocator(bundles, EDGE_CAPS, MEM_CAPS).solve()
    assert dict(result["selected"])["r1"] == "b0"


def test_all_baselines_respect_capacity():
    # Deliberately over-provisioned request set so infeasible choices are tempting.
    bundles = [
        _make_bundle("b0", "r1", 50.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
        _make_bundle("b1", "r1", 1.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 1, "B": 1}),
        _make_bundle("b0", "r2", 40.0, {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}),
    ]
    tight_edge_caps = {("A", "R"): 10, ("R", "B"): 10}
    tight_mem_caps = {"A": 10, "R": 10, "B": 10}

    results = run_all_baselines(bundles, tight_edge_caps, tight_mem_caps, seed=42)
    checker = FeasibilityChecker(tight_edge_caps, tight_mem_caps)
    by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}

    for name, result in results.items():
        feas = checker.check(result["selected"], by_key)
        assert feas["feasible"], f"{name} produced an infeasible selection"


def test_all_baselines_one_bundle_per_request():
    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 1}, {"A": 1, "R": 1}),
        _make_bundle("b1", "r1", 5.0, {("A", "R"): 1}, {"A": 1, "R": 1}),
        _make_bundle("b0", "r2", 8.0, {("R", "B"): 1}, {"R": 1, "B": 1}),
    ]
    results = run_all_baselines(bundles, EDGE_CAPS, MEM_CAPS, seed=1)
    for name, result in results.items():
        rids = [rid for rid, _ in result["selected"]]
        assert len(rids) == len(set(rids)), f"{name} selected more than one bundle for a request"


def test_highest_fidelity_first_prefers_fidelity_over_utility():
    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 1}, {"A": 1, "R": 1}, fidelity=0.6),
        _make_bundle("b1", "r1", 5.0, {("A", "R"): 1}, {"A": 1, "R": 1}, fidelity=0.95),
    ]
    result = HighestFidelityFirstAllocator(bundles, EDGE_CAPS, MEM_CAPS).solve()
    assert dict(result["selected"])["r1"] == "b1"


def test_congestion_aware_greedy_spreads_load_across_requests():
    # Two requests both want the same expensive bundle; only one can fit.
    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 6}, {"A": 6, "R": 6}),
        _make_bundle("b0", "r2", 10.0, {("A", "R"): 6}, {"A": 6, "R": 6}),
    ]
    tight_caps = {("A", "R"): 6, ("R", "B"): 6}
    result = CongestionAwareGreedyAllocator(bundles, tight_caps, MEM_CAPS).solve()
    assert len(result["selected"]) == 1


def test_greedy_local_search_never_worse_than_its_seed():
    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 3}, {"A": 3, "R": 3}),
        _make_bundle("b1", "r1", 12.0, {("A", "R"): 3}, {"A": 3, "R": 3}),
        _make_bundle("b0", "r2", 8.0, {("A", "R"): 3}, {"A": 3, "R": 3}),
    ]
    tight_caps = {("A", "R"): 6, ("R", "B"): 6}
    seed_result = UtilityPerResourceGreedyAllocator(bundles, tight_caps, MEM_CAPS).solve()
    ls_result = GreedyLocalSearchAllocator(bundles, tight_caps, MEM_CAPS, seed=0).solve()
    assert ls_result["total_utility"] >= seed_result["total_utility"]


def test_random_feasible_is_always_feasible_across_seeds():
    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 4}, {"A": 4, "R": 4}),
        _make_bundle("b1", "r1", 6.0, {("A", "R"): 2}, {"A": 2, "R": 2}),
    ]
    checker = FeasibilityChecker(EDGE_CAPS, MEM_CAPS)
    by_key = {(b["request_id"], b["bundle_id"]): b for b in bundles}
    for seed in range(10):
        result = RandomFeasibleAllocator(bundles, EDGE_CAPS, MEM_CAPS, seed=seed).solve()
        assert checker.check(result["selected"], by_key)["feasible"]


def test_compute_metrics_reports_optimality_gap():
    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 1}, {"A": 1, "R": 1}),
        _make_bundle("b1", "r1", 5.0, {("A", "R"): 1}, {"A": 1, "R": 1}),
    ]
    greedy_result = UtilityPerResourceGreedyAllocator(bundles, EDGE_CAPS, MEM_CAPS).solve()
    metrics = compute_metrics(greedy_result, bundles, EDGE_CAPS, MEM_CAPS, reference_utility=10.0)
    assert metrics["optimality_gap"] == 0.0
    assert metrics["feasible"] is True


def test_cp_sat_matches_or_beats_greedy_when_available():
    if not CPSAT_AVAILABLE:
        return  # optional dependency not installed in this environment
    from baselines.classical_baselines import CPSATAllocator

    bundles = [
        _make_bundle("b0", "r1", 10.0, {("A", "R"): 4}, {"A": 4, "R": 4}),
        _make_bundle("b1", "r1", 6.0, {("A", "R"): 2}, {"A": 2, "R": 2}),
        _make_bundle("b0", "r2", 9.0, {("A", "R"): 4}, {"A": 4, "R": 4}),
    ]
    tight_caps = {("A", "R"): 6, ("R", "B"): 6}
    greedy = UtilityPerResourceGreedyAllocator(bundles, tight_caps, MEM_CAPS).solve()
    exact = CPSATAllocator(bundles, tight_caps, MEM_CAPS).solve()
    assert exact["is_optimal"]
    assert exact["total_utility"] >= greedy["total_utility"]
