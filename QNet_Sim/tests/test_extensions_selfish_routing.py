import pytest

from extensions.selfish_routing import (
    CongestionGame,
    WardropNetwork,
    braess_capacity_sweep,
    braess_toll_sweep,
    braess_wardrop,
    pigou_instance,
    pigou_poa_sweep,
    toll_sweep,
)


def _pigou_bundles(n=3):
    return pigou_instance(n_players=n)


def _make_pigou_game(n=3, **kw):
    bundles, caps, ov = _pigou_bundles(n)
    game = CongestionGame(bundles, caps, demand_model="unit", seed=7)
    game._beta_override.update(ov["beta"])
    game._base_override.update(ov["base"])
    return game


def _assert_nash(game, selections):
    for rid in game.requests:
        loads_without = game.loads_without(selections, rid)
        current_cost = game.private_cost(rid, selections[rid], loads_without)
        for bid in game.strategies[rid]:
            dev_cost = (game.private_cost(rid, bid["bundle_id"],
                                          loads_without) + 1e-9)
            assert dev_cost >= current_cost - 1e-6, (
                f"{rid} improves by deviating from {selections[rid]} "
                f"to {bid['bundle_id']}: {dev_cost} < {current_cost}")


def test_latency_model_with_capacity_scaling():
    fast = tuple(sorted(("S", "T")))
    bundles, caps, _ = _pigou_bundles(1)
    game = CongestionGame(bundles, caps, demand_model="unit", beta=1.0)
    # beta scaled by 1/cap on the fast link
    assert game._beta_of(fast) == pytest.approx(1.0)


def test_unit_potential_decreases_under_best_response():
    game = _make_pigou_game(3)
    initial = {r: "R0_fast" for r in game.requests}
    prev = game.potential(initial)
    current = dict(initial)
    for _ in range(20):
        for rid in game.requests:
            new_bid, _ = game.best_response(rid, current)
            current[rid] = new_bid
        phi = game.potential(current)
        assert phi <= prev + 1e-9
        prev = phi
    _assert_nash(game, current)


def test_pigou_poa_values():
    for n, expected in [(2, 4 / 3), (3, 6 / 5), (4, 8 / 7), (8, 16 / 15)]:
        game = _make_pigou_game(n)
        equilibria = game.nash_equilibria(n_restarts=30)
        _, opt_sc = game.social_optimum()
        worst = max(game.social_cost(eq) for eq in equilibria)
        assert worst / opt_sc == pytest.approx(expected, rel=1e-6)
        for eq in equilibria:
            _assert_nash(game, eq)


def test_marginal_cost_toll_recovers_optimum_on_pigou():
    bundles, caps, ov = _pigou_bundles(3)
    rows = toll_sweep(bundles, caps, demand_model="unit",
                      beta_override=ov["beta"], base_override=ov["base"])
    untolled = next(r for r in rows if r["lambda"] == 0.0)
    tolled = next(r for r in rows if r["lambda"] == 1.0)
    assert untolled["poa_worst"] == pytest.approx(6 / 5)
    assert tolled["poa_worst"] <= 1.0 + 1e-9
    assert tolled["sc_worst"] == pytest.approx(tolled["opt_sc"])


def test_price_of_anarchy_at_least_one():
    game = _make_pigou_game(4)
    assert game.price_of_anarchy(n_restarts=10) >= 1.0 - 1e-9


def test_wardrop_braess_paradox():
    # Without the shortcut the equilibrium equals the social optimum.
    base = braess_wardrop(cross_base=1.0, add_cross=False)
    eq_base = base.equilibrium()
    assert eq_base["social_cost"] == pytest.approx(1.5)
    assert base.social_optimum() == pytest.approx(1.5)
    # A free shortcut raises the equilibrium latency toward 2.0 (paradox).
    net = braess_wardrop(cross_base=0.0, add_cross=True)
    eq = net.equilibrium()
    assert eq["social_cost"] == pytest.approx(2.0)
    assert net.social_optimum() == pytest.approx(1.5)
    # Classic price of anarchy bound 4/3.
    assert eq["social_cost"] / net.social_optimum() == pytest.approx(4 / 3)


def test_wardrop_marginal_toll_resolves_paradox():
    net = braess_wardrop(cross_base=0.0, add_cross=True)
    tolled = net.equilibrium(lam=1.0)
    assert tolled["social_cost"] == pytest.approx(1.5)
    assert net.price_of_anarchy(lam=1.0) <= 1.0 + 1e-9


def test_wardrop_flat_toll_on_shortcut():
    rows = braess_toll_sweep()
    untolled = next(r for r in rows if r["flat_toll"] == 0.0)
    heavy = rows[-1]
    assert untolled["poa_worst"] == pytest.approx(4 / 3)
    assert heavy["eq_social_cost"] == pytest.approx(heavy["opt_social_cost"])


def test_wardrop_capacity_sweep_shows_paradox():
    rows = braess_capacity_sweep()
    eqs = {r["cross_base_latency"]: r for r in rows}
    assert eqs[1.0]["eq_social_cost"] == pytest.approx(1.5)
    assert eqs[0.0]["eq_social_cost"] == pytest.approx(2.0)
    # Equilibrium social cost rises monotonically as the shortcut speeds up.
    cbs = [r["cross_base_latency"] for r in rows]
    scs = [r["eq_social_cost"] for r in rows]
    assert all(scs[i] <= scs[i + 1] + 1e-9 for i in range(len(scs) - 1))
    # Tolled equilibria stay at the optimum throughout.
    assert all(r["tolled_social_cost"] == pytest.approx(1.5) for r in rows)
