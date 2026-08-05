import random

import pytest

from baselines.qlearning_router import QLearningRouter


def _b(bid, rid, util, edge_d, mem_d, lat=0.0):
    return {
        "bundle_id": bid, "request_id": rid, "utility": util,
        "edge_demands": edge_d, "memory_demands": mem_d,
        "latency": lat, "path": ["A", "R", "B"],
    }


def _caps():
    return {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}


def test_state_quantization():
    ec, mc = _caps()
    r = QLearningRouter(ec, mc, n_bins=3, seed=0)
    assert r._state({}) == (0, 0)


def test_choose_action_prefers_feasible():
    ec, mc = _caps()
    r = QLearningRouter(ec, mc, epsilon=0.0, seed=0)
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
               _b("b1", "r1", 10.0, {("A", "R"): 12}, {"A": 12})]
    assert r.choose_action("r1", bundles, {}) == "b0"


def test_reject_when_infeasible():
    ec, mc = _caps()
    r = QLearningRouter(ec, mc, epsilon=0.0, seed=0)
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 12}, {"A": 12})]
    assert r.choose_action("r1", bundles, {}) is None


def test_train_episode_serves_feasible():
    ec, mc = _caps()
    r = QLearningRouter(ec, mc, seed=0)
    trace = [{"r1": [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1})]}]
    served, reward = r.train_episode(trace)
    assert served == 1
    assert reward == pytest.approx(50.0)


def test_evaluate_is_greedy_without_learning():
    ec, mc = _caps()
    r = QLearningRouter(ec, mc, seed=0, epsilon=0.5)
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
               _b("b1", "r1", 80.0, {("A", "R"): 1}, {"A": 1})]
    trace = [{"r1": bundles}]
    for _ in range(50):
        r.train_episode(trace)
    served, util, _ = r.evaluate(trace)
    assert served == 1
    assert util == pytest.approx(80.0)


def test_evaluate_does_not_mutate_q_table():
    ec, mc = _caps()
    r = QLearningRouter(ec, mc, seed=0)
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
               _b("b1", "r1", 80.0, {("A", "R"): 1}, {"A": 1})]
    trace = [{"r1": bundles}]
    r.train_episode(trace)
    before = len(r.q)
    r.evaluate(trace)
    assert len(r.q) == before
