from optimization.physics_hamiltonian import (
    PhysicalHamiltonian, UtilityTerm, OnePerRequestTerm,
    EdgeCapacityTerm, MemoryCapacityTerm, SoftCongestionTerm,
    MemoryRiskTerm,
)


def _b(bid, rid, util, edge_d, mem_d, lat=0.0):
    return {
        "bundle_id": bid, "request_id": rid, "utility": util,
        "edge_demands": edge_d, "memory_demands": mem_d,
        "latency": lat, "path": ["A", "R", "B"],
    }


def test_empty_hamiltonian():
    h = PhysicalHamiltonian([], {}, {})
    assert h.energy({}) == 0.0


def test_utility_term():
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1})]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 10}, {"A": 10})
    e0 = h.energy({("r1", "b0"): 0})
    e1 = h.energy({("r1", "b0"): 1})
    assert e1 == pytest.approx(-50.0, abs=1e-9)
    assert e0 > e1


def test_one_per_request():
    bundles = [
        _b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
        _b("b1", "r1", 30.0, {("A", "R"): 1}, {"A": 1}),
    ]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 10}, {"A": 10})
    e_both = h.energy({("r1", "b0"): 1, ("r1", "b1"): 1})
    e_one = h.energy({("r1", "b0"): 1, ("r1", "b1"): 0})
    assert e_both > e_one


def test_edge_capacity():
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 10}, {"A": 1})]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 5}, {"A": 10})
    e_viol = h.energy({("r1", "b0"): 1})
    e_ok = h.energy({("r1", "b0"): 0})
    assert e_viol > e_ok


def test_memory_capacity():
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 10})]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 10}, {"A": 5})
    e_viol = h.energy({("r1", "b0"): 1})
    e_ok = h.energy({("r1", "b0"): 0})
    assert e_viol > e_ok


def test_soft_congestion():
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 9}, {"A": 1})]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 10}, {"A": 10})
    h.clear_terms()
    h.add_term(UtilityTerm())
    h.add_term(SoftCongestionTerm(weight=10.0, warning_threshold=0.5))
    e = h.energy({("r1", "b0"): 1})
    assert e < 0
    cap = h.edge_capacities[("A", "R")]
    expected_cong = 10.0 * (9 / cap - 0.5) ** 2
    expected_total = -50.0 + expected_cong
    assert e == pytest.approx(expected_total, abs=1e-9)


def test_memory_risk():
    import math
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}, lat=10.0)]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 10}, {"A": 10})
    h.clear_terms()
    h.add_term(UtilityTerm())
    h.add_term(MemoryRiskTerm(weight=2.0, tau_mem=5.0))
    e = h.energy({("r1", "b0"): 1})
    expected_risk = 2.0 * (1.0 - math.exp(-10.0 / 5.0))
    assert e == pytest.approx(-50.0 + expected_risk, abs=1e-9)


def test_describe():
    bundles = [_b("b0", "r1", 10.0, {}, {})]
    h = PhysicalHamiltonian(bundles, {}, {})
    desc = h.describe()
    assert "utility" in desc.lower()
    assert "one_per_request" in desc.lower()


def test_custom_terms():
    bundles = [_b("b0", "r1", 10.0, {("A", "R"): 6}, {"A": 1})]
    h = PhysicalHamiltonian(bundles, {("A", "R"): 5}, {"A": 5})
    h.clear_terms()
    h.add_term(UtilityTerm(weight=2.0))
    h.add_term(EdgeCapacityTerm(weight=5.0, exponent=2.0))
    e_sel = h.energy({("r1", "b0"): 1})
    e_unsel = h.energy({("r1", "b0"): 0})
    assert e_sel == pytest.approx(-20.0 + 5.0 * (6 - 5) ** 2, abs=1e-9)
    assert e_unsel == 0.0


def test_slackfree_default_weights_anchor_to_utility_scale():
    bundles = [
        _b("b0", "r1", 200.0, {}, {}),
        _b("b1", "r1", 150.0, {}, {}),
    ]
    h = PhysicalHamiltonian(bundles, {}, {})
    q, offset = h.to_qubo_slackfree(utility_weight=1.0)  # no explicit weights
    pair = q.get(("x_0", "x_1"), q.get(("x_1", "x_0"), 0.0))
    assert pair > 200.0  # one-per-request weight exceeds the utility scale
    assert offset == 0.0


def test_decode():
    bundles = [
        _b("b0", "r1", 10.0, {}, {}),
        _b("b1", "r2", 20.0, {}, {}),
    ]
    h = PhysicalHamiltonian(bundles, {}, {})
    decoded = h.decode({"x_0": 1, "x_1": 0})
    assert decoded == [("r1", "b0")]


import pytest
