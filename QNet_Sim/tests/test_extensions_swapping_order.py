"""Tests for swapping-order strategy optimization (Extension 17)."""

import math

import pytest

from extensions.swapping_order import (
    swap_fidelity, linear_tree, balanced_tree, tree_depth, tree_peak_concurrency,
    evaluate_tree, linear_fidelity, strategy_metrics, strategy_fidelity,
    all_strategies, run_swapping_order_sweep, run_path_fidelity_sweep,
    strategy_frontier,
)
from fidelity.fidelity_model import FidelityModel


def _fids(n, base=0.85):
    return [base] * n


# ----------------------------------------------------------------------
# tree structure
# ----------------------------------------------------------------------
def test_tree_depth_and_concurrency():
    assert tree_depth(linear_tree(5)) == 4
    assert tree_depth(balanced_tree(5)) == 3
    # linear swaps all run one at a time
    assert tree_peak_concurrency(linear_tree(8)) == 1
    # balanced runs up to n/2 swaps in parallel
    assert tree_peak_concurrency(balanced_tree(8)) == 4


def test_evaluate_tree_matches_linear():
    fids = _fids(6, 0.87)
    assert evaluate_tree(linear_tree(6), fids) == pytest.approx(
        linear_fidelity(fids))
    # balanced tree gives the same noiseless fidelity (Werner associativity)
    assert evaluate_tree(balanced_tree(6), fids) == pytest.approx(
        linear_fidelity(fids), rel=1e-9)


def test_werner_swap_associativity():
    a, b, c = 0.82, 0.91, 0.77
    ab = swap_fidelity(a, b)
    assert swap_fidelity(swap_fidelity(a, b), c) == pytest.approx(
        swap_fidelity(a, swap_fidelity(b, c)), rel=1e-12)
    assert swap_fidelity(a, b) == pytest.approx(
        FidelityModel.entanglement_swapping(a, b))


# ----------------------------------------------------------------------
# strategies under coherence decay
# ----------------------------------------------------------------------
def test_strategy_fidelity_hold_times():
    fids = _fids(8)
    lin = strategy_fidelity(fids, "linear")
    bal = strategy_fidelity(fids, "balanced")
    opt = strategy_fidelity(fids, "optimal")
    assert lin["depth"] == 7
    assert bal["depth"] == 3
    assert opt["depth"] == bal["depth"]  # min-depth is optimal here
    assert lin["peak_concurrency"] == 1
    # noiseless fidelity identical across strategies
    assert lin["fidelity"] == pytest.approx(bal["fidelity"], rel=1e-9)
    assert lin["fidelity"] == pytest.approx(opt["fidelity"], rel=1e-9)


def test_decay_hurts_linear_more_than_balanced():
    fids = _fids(10, 0.85)
    tau = 3.0
    lin = strategy_fidelity(fids, "linear", delta=1.0, tau_mem=tau)
    bal = strategy_fidelity(fids, "balanced", delta=1.0, tau_mem=tau)
    assert bal["delivered_fidelity"] > lin["delivered_fidelity"]
    assert lin["hold_time"] == 9.0
    assert bal["hold_time"] == 4.0


def test_no_decay_when_tau_infinite():
    fids = _fids(6, 0.9)
    m = strategy_fidelity(fids, "linear", tau_mem=float("inf"))
    assert m["delivered_fidelity"] == pytest.approx(m["fidelity"])


def test_all_strategies_catalan_count():
    fids = _fids(4)
    # C_3 = 5 full binary trees over 4 leaves
    assert len(all_strategies(fids)) == 5


def test_all_strategies_capped_for_long_paths():
    # 9 leaves -> Catalan(8) = 1430 trees; exhaustive enumeration is skipped
    # and only the canonical linear/balanced trees are returned.
    rows = all_strategies(_fids(9))
    assert {r["strategy"] for r in rows} <= {"linear", "balanced"}
    assert len(rows) <= 2


def test_optimal_falls_back_to_balanced_for_long_paths():
    from extensions.swapping_order import MAX_EXHAUSTIVE_LEAVES
    n = MAX_EXHAUSTIVE_LEAVES + 2
    fids = _fids(n, 0.85)
    opt = strategy_fidelity(fids, "optimal", delta=1.0, tau_mem=3.0)
    bal = strategy_fidelity(fids, "balanced", delta=1.0, tau_mem=3.0)
    assert opt["depth"] == bal["depth"] == math.ceil(math.log2(n))
    assert opt["delivered_fidelity"] == pytest.approx(
        bal["delivered_fidelity"], rel=1e-12)


def test_fidelity_inputs_are_validated():
    with pytest.raises(ValueError):
        FidelityModel.end_to_end_fidelity([0.85, 1.5])
    with pytest.raises(ValueError):
        FidelityModel.purification_bbpssw(2.0)
    with pytest.raises(ValueError):
        FidelityModel.purification_bbpssw(0.4)  # below the purification floor
    with pytest.raises(ValueError):
        FidelityModel.entanglement_swapping(0.85, -0.1)


def test_purification_floor_is_explicit():
    # at the 0.5 floor the BBPSSW map is the identity
    assert FidelityModel.purification_bbpssw(0.5) == pytest.approx(0.5, abs=1e-12)
    # strictly above the floor it improves
    assert FidelityModel.purification_bbpssw(0.7) > 0.7


def test_strategy_frontier_non_dominated():
    fids = _fids(6, 0.85)
    front = strategy_frontier(fids, delta=1.0, tau_mem=3.0)
    assert front
    for i, p in enumerate(front):
        for j, q in enumerate(front):
            if i == j:
                continue
            better = (q["depth"] <= p["depth"]
                      and q["peak_concurrency"] <= p["peak_concurrency"]
                      and q["delivered_fidelity"] >= p["delivered_fidelity"] - 1e-12)
            strict = (q["depth"] < p["depth"]
                      or q["peak_concurrency"] < p["peak_concurrency"]
                      or q["delivered_fidelity"] > p["delivered_fidelity"] + 1e-12)
            assert not (better and strict), f"{p} dominated by {q}"


# ----------------------------------------------------------------------
# experiment helpers
# ----------------------------------------------------------------------
def test_run_swapping_order_sweep_rows():
    rows = run_swapping_order_sweep(path_lengths=[3, 4], n_trials=5, seed=42)
    assert len(rows) == 2 * 5 * 3  # lengths x trials x strategies
    for r in rows:
        assert r["strategy"] in ("linear", "balanced", "optimal")
        assert r["path_length"] in (3, 4)
        assert r["delivered_fidelity"] <= r["fidelity"] + 1e-12
        assert r["depth"] == r["path_length"] - 1 or r["depth"] < r["path_length"] - 1


def test_run_path_fidelity_sweep_trend():
    rows = run_path_fidelity_sweep(path_length=8, link_fidelity=0.85,
                                   tau_mem_values=[1.0, 3.0, 10.0])
    by = {(r["strategy"], r["tau_mem"]): r["delivered_fidelity"] for r in rows}
    # linear suffers most: at every tau, optimal >= balanced >= linear
    for tau in (1.0, 3.0, 10.0):
        assert by[("optimal", tau)] >= by[("balanced", tau)] - 1e-12
        assert by[("balanced", tau)] >= by[("linear", tau)] - 1e-12
    # more coherence (larger tau) never hurts any strategy
    for strat in ("linear", "balanced", "optimal"):
        assert by[(strat, 10.0)] > by[(strat, 1.0)]
