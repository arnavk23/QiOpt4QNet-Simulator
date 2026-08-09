import random

import pytest

from extensions.robust_routing import (
    RobustRoutingModel,
    bundle_key,
    generate_scenarios,
)


def _fragile_instance(seed=3, failure_prob=0.6, severity=0.9, n_scenarios=12):
    """Two requests compete for one fragile bottleneck edge."""
    bottleneck = tuple(sorted(("S", "B")))

    def mk(rid, bid, edges, util):
        return {"bundle_id": bid, "request_id": rid, "path": list(edges),
                "edge_demands": {tuple(sorted(e)): 1 for e in edges},
                "memory_demands": {}, "utility": util}

    bundles = [
        mk("R1", "high", [("S", "B"), ("B", "T1")], 100.0),
        mk("R1", "safe", [("S", "T1")], 60.0),
        mk("R2", "high", [("S", "B"), ("B", "T2")], 95.0),
        mk("R2", "safe", [("S", "T2")], 50.0),
    ]
    caps = {
        bottleneck: 1.0,
        tuple(sorted(("B", "T1"))): 10.0,
        tuple(sorted(("B", "T2"))): 10.0,
        tuple(sorted(("S", "T1"))): 10.0,
        tuple(sorted(("S", "T2"))): 10.0,
    }
    mem = {"S": 100.0, "B": 100.0, "T1": 100.0, "T2": 100.0}
    rng = random.Random(seed)
    scenarios = []
    for _ in range(n_scenarios):
        fail = rng.random() < failure_prob
        scenario = {}
        for b in bundles:
            u = b["utility"] * (1.0 - severity) if fail and bottleneck in b["edge_demands"] else b["utility"]
            scenario[bundle_key(b)] = u
        scenarios.append(scenario)
    return bundles, caps, mem, scenarios


def test_generate_scenarios_shape_and_nominal_first():
    bundles, caps, mem, _ = _fragile_instance()
    scenarios = generate_scenarios(bundles, n_scenarios=6, noise_scale=0.1, seed=1)
    assert len(scenarios) == 6
    nominal = scenarios[0]
    for b in bundles:
        assert nominal[bundle_key(b)] == pytest.approx(b["utility"])
    for s in scenarios[1:]:
        assert set(s.keys()) == set(nominal.keys())


def test_robust_coefficient_endpoints():
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=1)
    key = ("R1", "high")
    assert model.robust_coefficient(key, 0.0) == pytest.approx(model.nominal[key])
    assert model.robust_coefficient(key, 1.0) == pytest.approx(
        min(s[key] for s in scenarios))
    # robust coefficient is monotone non-increasing in gamma
    assert model.robust_coefficient(key, 0.5) <= model.nominal[key] + 1e-9
    assert model.robust_coefficient(key, 1.0) <= model.robust_coefficient(key, 0.5) + 1e-9


def test_objective_modes():
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=1)
    selections = {"R1": "safe", "R2": "high"}
    nominal = model.objective(selections, "nominal")
    worst = model.objective(selections, "worst")
    mean = model.objective(selections, "mean")
    assert nominal == pytest.approx(60.0 + 95.0)
    assert worst == pytest.approx(60.0 + 95.0 * (1.0 - 0.9))
    assert mean >= worst and mean <= nominal


def test_exact_robustness_improves_worst_case():
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=42)
    nominal = model.solve_exact(0.0)
    robust = model.solve_exact(1.0)
    ev_n = model.evaluate(nominal)
    ev_r = model.evaluate(robust)
    # Nominal routing is better in expectation, robust routing far better in
    # the worst scenario.
    assert ev_n["nominal_util"] > ev_r["nominal_util"]
    assert ev_r["worst_util"] > ev_n["worst_util"]


def test_pareto_sweep_tradeoff_exact():
    import itertools
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=42)
    rows = model.pareto_sweep(gammas=[0.0, 0.5, 1.0], exact=True)
    assert rows[0]["gamma"] == 0.0 and rows[-1]["gamma"] == 1.0
    # worst-case utility improves with the robustness budget
    assert rows[2]["worst_util"] >= rows[0]["worst_util"] - 1e-9

    # The gamma=1 solution achieves the best possible worst-case utility and
    # the gamma=0 solution the best possible nominal utility (brute force).
    groups = {}
    for b in bundles:
        groups.setdefault(b["request_id"], []).append(b)
    best_worst = -float("inf")
    best_nominal = -float("inf")
    for combo in itertools.product(*groups.values()):
        sel = {b["request_id"]: b["bundle_id"] for b in combo}
        if not model._feasible(sel):
            continue
        best_worst = max(best_worst, min(sum(s[k] for k in sel.items() if k[1] is not None)
                                         for s in scenarios))
        best_nominal = max(best_nominal, model.objective(sel, "nominal"))
    assert rows[2]["worst_util"] == pytest.approx(best_worst)
    assert rows[0]["nominal_util"] == pytest.approx(best_nominal)


def test_robustness_gain_exact():
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=42)
    gain = model.robustness_gain(exact=True)
    # maximin routing raises the worst-case utility (its defining objective)
    assert gain["worst_util_gain"] > 0
    assert gain["nominal_util_loss"] > 0


def test_min_max_regret_is_separate_from_maximin():
    import itertools
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=42)
    regret_opt = model.solve_exact_regret()

    # The regret minimiser attains the minimum possible max-regret (brute force).
    hindsight = {id(s): model._exact_best_per_scenario(s) for s in scenarios}
    best_max_regret = float("inf")
    groups = {}
    for b in bundles:
        groups.setdefault(b["request_id"], []).append(b)
    for combo in itertools.product(*groups.values()):
        sel = {b["request_id"]: b["bundle_id"] for b in combo}
        if not model._feasible(sel):
            continue
        mx = max(h - sum(s.get((rid, bid), 0.0) for rid, bid in sel.items() if bid is not None)
                 for s, h in zip(scenarios, hindsight.values()))
        best_max_regret = min(best_max_regret, mx)
    reg_o = model.regret(regret_opt, exact=True)
    assert reg_o["max_regret"] == pytest.approx(best_max_regret)


def test_solve_matches_exact_on_tiny_instance():
    # With a big enough annealing budget the heuristic should approach the
    # exact optimum on the tiny fragile instance (gamma = 0).
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=42)
    exact = model.solve_exact(1.0)
    approx = model.solve(1.0, penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                         max_iterations=8000, n_restarts=6, steps_per_temperature=10)
    assert model.objective(approx, "robust", gamma=1.0) >= model.objective(exact, "robust", gamma=1.0) - 1e-9


def test_feasibility_checker():
    bundles, caps, mem, scenarios = _fragile_instance()
    model = RobustRoutingModel(bundles, caps, mem, scenarios, seed=1)
    # both high bundles exceed the bottleneck capacity -> infeasible
    assert not model._feasible({"R1": "high", "R2": "high"})
    # one high + one safe is feasible
    assert model._feasible({"R1": "high", "R2": "safe"})
