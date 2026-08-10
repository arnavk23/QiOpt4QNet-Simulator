"""Tests for chance-constrained fidelity routing."""

import math
import random

import pytest

from optimization.chance_constrained import (
    expand_instance, filter_policy, hard_feasible, chance_feasible,
    sample_truncnorm, truncnorm_cdf, violation_probability,
    run_chance_constrained_study,
)
from experiments.instances import generate_chain_topology


def _topo():
    return generate_chain_topology(n_nodes=8, edge_capacity=6,
                                   memory_capacity=10, raw_fidelity=0.85,
                                   generation_prob=0.8)


def _pairs(n_req=4, seed=3):
    rng = random.Random(seed)
    topo = _topo()
    pairs = []
    for _ in range(n_req):
        s, d = rng.sample(topo["nodes"], 2)
        pairs.append((s, d, rng.uniform(10.0, 100.0), rng.uniform(0.5, 0.7)))
    return pairs


# ----------------------------------------------------------------------
# truncated-normal fidelity model
# ----------------------------------------------------------------------
def test_truncnorm_cdf_limits_and_monotonicity():
    # mean above requirement -> near-certain compliance
    assert truncnorm_cdf(0.8, mu=0.85, sigma=0.01) < 1e-3
    # requirement equals mean -> ~1/2 under symmetric noise
    assert abs(truncnorm_cdf(0.8, mu=0.8, sigma=0.01) - 0.5) < 0.05
    # monotone in the requirement
    assert truncnorm_cdf(0.79, mu=0.8, sigma=0.01) < truncnorm_cdf(0.81, mu=0.8, sigma=0.01)
    # degenerate sigma recovers a step function
    assert truncnorm_cdf(0.79, mu=0.8, sigma=0.0) == 0.0
    assert truncnorm_cdf(0.81, mu=0.8, sigma=0.0) == 1.0


def test_violation_probability_bounds():
    p = violation_probability(0.80, 0.80, 0.01)
    assert 0.4 < p < 0.6
    p_low = violation_probability(0.85, 0.80, 0.01)
    assert p_low < 1e-3
    p_high = violation_probability(0.75, 0.80, 0.01)
    assert p_high > 0.999


def test_chance_feasible_reduces_to_hard_at_zero_sigma():
    for f in [0.79, 0.80, 0.81]:
        for mf in [0.79, 0.80, 0.81]:
            assert chance_feasible(f, mf, eps=0.05, sigma=0.0) == (f >= mf)


def test_chance_feasible_eps_monotonic():
    mu, mf, sig = 0.795, 0.80, 0.005
    feats = [chance_feasible(mu, mf, eps, sig) for eps in [0.001, 0.01, 0.1, 0.5]]
    assert feats == sorted(feats)
    assert chance_feasible(mu, mf, eps=1.0, sigma=sig) is True


def test_sample_truncnorm_bounds_and_determinism():
    rng1, rng2 = random.Random(7), random.Random(7)
    a = [sample_truncnorm(0.8, 0.01, rng1) for _ in range(500)]
    b = [sample_truncnorm(0.8, 0.01, rng2) for _ in range(500)]
    assert a == b
    assert all(0.0 <= x <= 1.0 for x in a)
    assert abs(sum(a) / len(a) - 0.8) < 0.02


# ----------------------------------------------------------------------
# candidate expansion
# ----------------------------------------------------------------------
def test_expand_instance_keeps_near_miss_bundles():
    bundles = expand_instance(_topo(), _pairs(n_req=4), sigma=0.01)
    near_miss = [b for b in bundles
                 if b["fidelity"] < b["min_fidelity"]
                 and b["fidelity"] >= 0.3]
    assert near_miss, "expanded set must contain F < F_min candidates"
    for b in bundles:
        assert "min_fidelity" in b and "weight" in b and "p_viol" in b
        assert 0.0 <= b["p_viol"] <= 1.0


def test_expand_instance_utility_uses_true_min_fidelity():
    pairs = _pairs(n_req=2)
    bundles = expand_instance(_topo(), pairs, sigma=0.01)
    for b in bundles:
        mf = b["min_fidelity"]
        w = b["weight"]
        margin = max(0.0, b["fidelity"] - mf)
        expected = (w * b["success_probability"] * (1.0 + margin)
                    - 0.01 * b["latency"] - 0.05 * b["bell_pair_cost"])
        assert math.isclose(b["utility"], expected, rel_tol=1e-9)


def _key(b):
    return (b["request_id"], b["bundle_id"])


def test_filter_policy_zero_sigma_matches_hard():
    bundles = expand_instance(_topo(), _pairs(n_req=4), sigma=0.0)
    hard = set(_key(b) for b in filter_policy(bundles, "hard"))
    for eps in [0.05, 0.2, 0.5]:
        chance = set(_key(b) for b in filter_policy(bundles, "chance", eps=eps))
        assert chance == hard


def test_chance_above_half_admits_near_miss_that_hard_refuses():
    # Under symmetric noise a bundle whose mean sits just below the
    # requirement violates more than half the time, so hard rejects it.
    # chance(eps>1/2) is the only regime that admits the near-miss.
    topo = _topo()
    bundles = near_miss = None
    for seed in range(8):
        bundles = expand_instance(topo, _pairs(n_req=6, seed=seed), sigma=0.03)
        near_miss = next((b for b in bundles
                          if 0.0 < b["min_fidelity"] - b["fidelity"] <= 0.02),
                         None)
        if near_miss is not None:
            break
    if near_miss is None:
        pytest.skip("no near-miss bundle in this seed")
    key = _key(near_miss)
    hard = {_key(b) for b in filter_policy(bundles, "hard")}
    assert key not in hard
    assert near_miss["p_viol"] > 0.5
    loose_eps = max(0.6, near_miss["p_viol"] + 1e-9)
    loose = {_key(b) for b in filter_policy(bundles, "chance", eps=loose_eps)}
    assert key in loose


def test_chance_below_half_requires_safety_margin_over_hard():
    # chance(eps<1/2) demands F >= F_min + z_eps*sigma, i.e. it is strictly
    # stricter than hard: it refuses every bundle hard also refuses, and
    # additionally prunes any bundle whose margin is under z_eps*sigma.
    bundles = expand_instance(_topo(), _pairs(n_req=6), sigma=0.04)
    hard = {_key(b) for b in filter_policy(bundles, "hard")}
    tight = {_key(b) for b in filter_policy(bundles, "chance", eps=0.1)}
    assert tight <= hard


# ----------------------------------------------------------------------
# exact frontier
# ----------------------------------------------------------------------
def test_exact_frontier_nominal_is_upper_bound():
    from experiments.optimality_benchmark import exact_ilp_solution
    ec, mc = _topo()["edge_capacities"], _topo()["memory_capacities"]

    bundles = expand_instance(_topo(), _pairs(n_req=4), sigma=0.01)
    u_hard = exact_ilp_solution(filter_policy(bundles, "hard"),
                                ec, mc)["u_star"]
    u_chance = exact_ilp_solution(filter_policy(bundles, "chance", eps=0.1),
                                  ec, mc)["u_star"]
    u_nom = exact_ilp_solution(filter_policy(bundles, "nominal"),
                               ec, mc)["u_star"]
    assert u_hard <= u_nom + 1e-6
    assert u_chance <= u_nom + 1e-6


def test_exact_frontier_zero_sigma_chance_equals_hard():
    from experiments.optimality_benchmark import exact_ilp_solution
    ec, mc = _topo()["edge_capacities"], _topo()["memory_capacities"]
    bundles = expand_instance(_topo(), _pairs(n_req=4), sigma=0.0)
    u_hard = exact_ilp_solution(filter_policy(bundles, "hard"),
                                ec, mc)["u_star"]
    u_chance = exact_ilp_solution(filter_policy(bundles, "chance", eps=0.1),
                                  ec, mc)["u_star"]
    assert math.isclose(u_hard, u_chance, rel_tol=1e-6)


# ----------------------------------------------------------------------
# DES fidelity noise
# ----------------------------------------------------------------------
def test_des_fidelity_noise_widens_fidelity_spread():
    from simulation.discrete_event_engine import StochasticEventSimulator
    topo = _topo()
    bundles = expand_instance(topo, _pairs(n_req=4), sigma=0.0)
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    from experiments.optimality_benchmark import exact_ilp_solution
    sel = exact_ilp_solution(bundles, ec, mc)["selected"]
    quiet = StochasticEventSimulator(topo, tau_mem=50.0, seed=42)
    noisy = StochasticEventSimulator(topo, tau_mem=50.0, seed=42,
                                     fidelity_noise_sigma=0.02)
    rq = quiet.simulate_plan(bundles, sel, n_realizations=60)
    rn = noisy.simulate_plan(bundles, sel, n_realizations=60)
    spread_q = rq["fid_q95"] - rq["fid_q05"]
    spread_n = rn["fid_q95"] - rn["fid_q05"]
    assert spread_n > spread_q
    # noise is unbiased on the delivered-fidelity mean
    assert abs(rn["e_delivered_fidelity"] - rq["e_delivered_fidelity"]) < 0.05


def test_des_fidelity_noise_deterministic():
    from simulation.discrete_event_engine import StochasticEventSimulator
    topo = _topo()
    bundles = expand_instance(topo, _pairs(n_req=4), sigma=0.0)
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    from experiments.optimality_benchmark import exact_ilp_solution
    sel = exact_ilp_solution(bundles, ec, mc)["selected"]
    a = StochasticEventSimulator(topo, seed=1, fidelity_noise_sigma=0.02)
    b = StochasticEventSimulator(topo, seed=1, fidelity_noise_sigma=0.02)
    assert (a.simulate_plan(bundles, sel, n_realizations=30)
            == b.simulate_plan(bundles, sel, n_realizations=30))


# ----------------------------------------------------------------------
# study runner
# ----------------------------------------------------------------------
def test_run_chance_constrained_study_schema():
    topo_fn = lambda: _topo()
    res = run_chance_constrained_study(
        topo_fn, n_requests_list=[4], eps_list=[0.1, 0.3, 0.6],
        sigma_list=[0.01], n_instances=1, n_realizations=15,
        time_limit=20.0, seed=42)
    rows = res["rows"]
    assert rows
    policies = {r["policy"] for r in rows}
    assert {"hard", "chance", "nominal", "chance_qubo"} <= policies
    for r in rows:
        assert 0.0 <= r["sla_violation_prob"] <= 1.0
        assert r["param_utility"] >= 0.0
        assert r["n_selected"] >= 0


def test_study_chance_sla_capped_by_eps():
    # The empirical SLA of a chance(eps) plan must stay at or below eps
    # (up to Monte-Carlo sampling noise): the chance rule is calibrated,
    # unlike hard whose implicit risk depends on the margin distribution.
    res = run_chance_constrained_study(
        lambda: _topo(), n_requests_list=[4], eps_list=[0.1, 0.3],
        sigma_list=[0.03], n_instances=1, n_realizations=100,
        time_limit=20.0, seed=42)
    chance_rows = [r for r in res["rows"] if r["policy"] == "chance"]
    assert chance_rows
    for r in chance_rows:
        assert r["sla_violation_prob"] <= r["eps"] + 0.05


def test_qubo_on_chance_set_is_feasible():
    from experiments.optimality_benchmark import exact_ilp_solution
    bundles = expand_instance(_topo(), _pairs(n_req=5), sigma=0.01)
    ec, mc = _topo()["edge_capacities"], _topo()["memory_capacities"]
    feasible = filter_policy(bundles, "chance", eps=0.1)
    qr = _qubo_import()
    r = qr(feasible, ec, mc, k=8, num_reads=10, seed=1)
    # decode is a valid selection (may be empty, but never malformed)
    assert isinstance(r.get("selected", []), list)
    exact = exact_ilp_solution(feasible, ec, mc, time_limit=30.0)
    assert exact["u_star"] is not None
    qutil = sum(b["utility"] for b in feasible
                if (b["request_id"], b["bundle_id"]) in
                set(r.get("selected", [])))
    assert qutil <= exact["u_star"] + 1e-6


def _qubo_import():
    from optimization.chance_constrained import qubo_solve
    return qubo_solve
