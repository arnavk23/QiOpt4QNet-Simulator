import math

import pytest

from optimization.conventional_calibrator import (
    conventional_coefficients,
    penalty_epsilon,
)
from optimization.openjij_solver import calibrated_coefficients
from optimization.proposed_calibrator import (
    coefficient_bound,
    possible_loads,
    proposed_global_coefficients,
)
from optimization.qubo_optimizer import QUBOOptimizer


def _bundle(bundle_id, request_id, utility, edge_demand=0, memory_demand=0):
    return {
        "bundle_id": bundle_id,
        "request_id": request_id,
        "path": ["A", "B"],
        "edge_demands": {("A", "B"): edge_demand} if edge_demand else {},
        "memory_demands": {"A": memory_demand} if memory_demand else {},
        "utility": utility,
    }


def _optimizer(bundles, edge_capacity=10, memory_capacity=10):
    return QUBOOptimizer(
        bundles,
        {("A", "B"): edge_capacity},
        {"A": memory_capacity},
    )


def test_penalty_epsilon_is_positive():
    assert penalty_epsilon(0.0) > 0.0
    assert penalty_epsilon(100.0) > penalty_epsilon(0.0)


def test_conventional_uses_largest_positive_utility():
    opt = _optimizer(
        [
            _bundle("b0", "r0", -5.0, 1, 1),
            _bundle("b1", "r1", 12.0, 1, 1),
            _bundle("b2", "r2", 7.0, 1, 1),
        ]
    )
    coeffs = conventional_coefficients(opt)
    assert coeffs["A"] > 12.0
    assert coeffs["A"] == coeffs["B"] == coeffs["D"]
    assert coeffs["C"] == 0.0
    assert coeffs["E"] == 0.0


def test_possible_loads_respects_one_choice_per_request():
    pairs = [
        (("r1", "a"), 2),
        (("r1", "b"), 4),
        (("r2", "a"), 3),
    ]
    loads = possible_loads(pairs, "excluded")
    # r1 contributes 0,2,or4 and r2 contributes 0 or 3.
    assert loads == {0, 2, 3, 4, 5, 7}


def test_possible_loads_excludes_candidate_request():
    pairs = [
        (("r1", "a"), 4),
        (("r2", "a"), 3),
    ]
    assert possible_loads(pairs, "r1") == {0, 3}


def test_resource_aware_edge_bound_can_be_smaller_than_conventional():
    # Two 4-unit bundles on capacity 5 produce overload 3 when combined.
    # Removing one reduces squared overload by 9, so B need not be as large
    # as the full utility scale.
    opt = _optimizer(
        [
            _bundle("b0", "r0", 18.0, edge_demand=4),
            _bundle("b1", "r1", 10.0, edge_demand=4),
        ],
        edge_capacity=5,
        memory_capacity=100,
    )
    conventional = conventional_coefficients(opt)
    proposed = proposed_global_coefficients(opt)

    assert proposed["A"] == conventional["A"]
    assert proposed["B"] < conventional["B"]
    assert proposed["B"] > 18.0 / 9.0


def test_no_possible_overload_needs_only_epsilon_resource_penalty():
    opt = _optimizer(
        [
            _bundle("b0", "r0", 20.0, edge_demand=2),
            _bundle("b1", "r1", 10.0, edge_demand=2),
        ],
        edge_capacity=4,
        memory_capacity=100,
    )
    proposed = proposed_global_coefficients(opt)
    assert math.isclose(proposed["B"], penalty_epsilon(20.0))


def test_memory_is_calibrated_separately_from_edges():
    opt = _optimizer(
        [
            _bundle("b0", "r0", 16.0, edge_demand=1, memory_demand=4),
            _bundle("b1", "r1", 8.0, edge_demand=1, memory_demand=4),
        ],
        edge_capacity=100,
        memory_capacity=5,
    )
    proposed = proposed_global_coefficients(opt)
    assert proposed["B"] < 1e-3
    assert proposed["D"] > 16.0 / 9.0


def test_coefficient_bound_uses_reachable_loads_only():
    grouped = {
        "edge": [
            (("r0", "b0"), 4),
            (("r1", "b0"), 1),
            (("r1", "b1"), 2),
        ]
    }
    capacities = {"edge": 5}
    utilities = {
        ("r0", "b0"): 9.0,
        ("r1", "b0"): 1.0,
        ("r1", "b1"): 1.0,
    }
    # For r0, the only competing loads are 0,1,2; only load 2 violates with d=4.
    # Delta=(2+4-5)^2=1, so bound must reach 9.
    assert coefficient_bound(grouped, capacities, utilities) >= 9.0


def test_calibrated_api_scales_only_hard_coefficients():
    opt = _optimizer(
        [
            _bundle("b0", "r0", 10.0, 4, 4),
            _bundle("b1", "r1", 8.0, 4, 4),
        ],
        edge_capacity=5,
        memory_capacity=5,
    )
    base = calibrated_coefficients(
        opt,
        "resource_aware",
        coefficient_scale=1.0,
        congestion_penalty=0.25,
        memory_congestion_penalty=0.5,
    )
    scaled = calibrated_coefficients(
        opt,
        "resource_aware",
        coefficient_scale=2.0,
        congestion_penalty=0.25,
        memory_congestion_penalty=0.5,
    )

    for key in ("A", "B", "D"):
        assert math.isclose(scaled[key], 2.0 * base[key])
    assert scaled["C"] == base["C"] == 0.25
    assert scaled["E"] == base["E"] == 0.5


def test_fixed_strategy_requires_coefficients():
    opt = _optimizer([_bundle("b0", "r0", 1.0, 1, 1)])
    with pytest.raises(ValueError, match="fixed_coefficients"):
        calibrated_coefficients(opt, "fixed")


def test_unknown_strategy_rejected():
    opt = _optimizer([_bundle("b0", "r0", 1.0, 1, 1)])
    with pytest.raises(ValueError, match="strategy"):
        calibrated_coefficients(opt, "mystery")