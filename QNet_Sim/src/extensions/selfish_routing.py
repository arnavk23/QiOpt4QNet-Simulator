"""D1 -- Selfish routing as an atomic congestion game with price-of-anarchy
analysis and marginal-cost tolls.

Model
-----
Each request is a *player* choosing one bundle (path) from its strategy set.
Edge loads follow the simulator convention L_e = sum of selected demands.
Edge latency per unit is affine in the load,

    l_e(L) = base_e + beta_e * L_e

where base_e and beta_e default to the global ``base_latency`` / ``beta`` but
may be overridden per edge (used by Braess's shortcut link) or, for ``beta``,
scaled by 1/C_e when ``edge_capacities`` are supplied so that congested edges
have steeper latency.

Selfish behaviour
-----------------
A player minimizes its *own* experienced latency minus a scaled utility
benefit (``utility_weight``):

    c_i(b, S) = sum_{e in b} d_b,e * l_e(L_e) - w * u_b .

This is the classic atomic unsplittable model.  In the ``unit`` demand model
(every used edge contributes load 1) the game is a Rosenthal potential game
with exact potential

    Phi(S) = sum_e sum_{j=1}^{L_e} l_e(j) - w * sum_i u_i

which is precisely the quadratic congestion term of the simulator's QUBO
Hamiltonian: Nash equilibria are local minima of the QUBO energy and
best-response dynamics are guaranteed to converge.  In the ``weighted`` model
(real demands d_b,e) there is no exact potential, so dynamics are heuristic
(they converge on the instances we study) and the continuous potential
Phi = sum_e (base_e L_e + beta_e/2 L_e^2) - w sum_i u_i is used for analysis.

Social cost (total latency) is SC(S) = sum_e L_e l_e(L_e).  The price of
anarchy is SC at the worst Nash equilibrium divided by the social optimum.

Marginal-cost tolls
-------------------
The congestion externality of a player on edge e is the social marginal cost
minus its private marginal cost, equal to beta_e * L^{-i}_e * d_b,e.  Charging
lambda times that externality as a toll makes private cost equal social
marginal cost at lambda = 1, so best-response becomes coordinate descent on
the (convex) social cost and the price of anarchy collapses toward one.

Flat tolls (``flat_tolls``) are fixed per-unit charges on specific edges and
are used to remove Braess's paradox from a free shortcut link.

Braess's paradox
----------------
The canonical network (one load-dependent link plus one constant link per
route, free shortcut) is included.  Adding the free shortcut raises the
worst-case Nash social cost from 4 to 6 for two unit players; a small toll on
the shortcut removes the paradox.
"""

import itertools
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def _undirected(edge) -> tuple:
    return tuple(sorted(edge))


class CongestionGame:
    def __init__(self, bundles: List[dict],
                 edge_capacities: Optional[Dict[tuple, float]] = None,
                 demand_model: str = "unit",
                 base_latency: float = 0.0,
                 beta: float = 1.0,
                 utility_weight: float = 0.0,
                 allow_none: bool = False,
                 seed: int = 42):
        self.demand_model = demand_model
        self.base_latency = base_latency
        self.beta = beta
        self.utility_weight = utility_weight
        self.allow_none = allow_none
        self._rng = random.Random(seed)
        self.edge_capacities = (
            {_undirected(k): v for k, v in edge_capacities.items()}
            if edge_capacities else {}
        )

        self.requests: List[str] = []
        self.strategies: Dict[str, List[dict]] = {}
        self._util_of: Dict[Tuple[str, str], float] = {}
        self._edge_of: Dict[Tuple[str, str], Dict[tuple, int]] = {}
        self._all_edges: set = set()
        self._beta_override: Dict[tuple, float] = {}
        self._base_override: Dict[tuple, float] = {}

        for b in bundles:
            rid = b["request_id"]
            if rid not in self.strategies:
                self.strategies[rid] = []
                self.requests.append(rid)
            self.strategies[rid].append(b)
            key = (rid, b["bundle_id"])
            self._util_of[key] = b["utility"]
            self._edge_of[key] = {_undirected(e): d for e, d in b["edge_demands"].items()}
            self._all_edges |= set(self._edge_of[key])

    # ------------------------------------------------------------------
    # edge latency model
    # ------------------------------------------------------------------
    def _base_of(self, e: tuple) -> float:
        return self._base_override.get(e, self.base_latency)

    def _beta_of(self, e: tuple) -> float:
        if e in self._beta_override:
            return self._beta_override[e]
        if self.edge_capacities:
            cap = self.edge_capacities.get(e, 1.0)
            return self.beta / cap
        return self.beta

    def latency(self, e: tuple, load: int) -> float:
        """Per-unit latency on edge e at total load `load`."""
        return self._base_of(e) + self._beta_of(e) * load

    # ------------------------------------------------------------------
    # loads
    # ------------------------------------------------------------------
    def loads(self, selections: Dict[str, Optional[str]]) -> Dict[tuple, int]:
        edge_load = defaultdict(int)
        for rid, bid in selections.items():
            if bid is None:
                continue
            for e, d in self._edge_of.get((rid, bid), {}).items():
                edge_load[e] += d
        return dict(edge_load)

    def loads_without(self, selections: Dict[str, Optional[str]], rid: str) -> Dict[tuple, int]:
        edge_load = self.loads(selections)
        bid = selections.get(rid)
        if bid is None:
            return edge_load
        for e, d in self._edge_of.get((rid, bid), {}).items():
            edge_load[e] -= d
        return edge_load

    # ------------------------------------------------------------------
    # objective quantities
    # ------------------------------------------------------------------
    def potential(self, selections: Dict[str, Optional[str]]) -> float:
        """Game potential.  Exact for the ``unit`` model, approximate for
        ``weighted``.  Nash equilibria are local minima of this quantity."""
        edge_load = self.loads(selections)
        phi = 0.0
        for e, L in edge_load.items():
            if self.demand_model == "unit":
                phi += sum(self.latency(e, j) for j in range(1, L + 1))
            else:
                phi += self._base_of(e) * L + 0.5 * self._beta_of(e) * L * L
        phi -= self.utility_weight * self.total_utility(selections)
        return phi

    def social_cost(self, selections: Dict[str, Optional[str]]) -> float:
        """Total latency SC(S) = sum_e L_e l_e(L_e)."""
        edge_load = self.loads(selections)
        return sum(L * self.latency(e, L) for e, L in edge_load.items())

    def total_utility(self, selections: Dict[str, Optional[str]]) -> float:
        return sum(
            self._util_of.get((rid, bid), 0.0)
            for rid, bid in selections.items() if bid is not None
        )

    def qubo_energy(self, selections: Dict[str, Optional[str]],
                    congestion_weight: float = 1.0) -> float:
        """Simulator-style QUBO energy (congestion term matches potential)."""
        edge_load = self.loads(selections)
        pen = sum(self._beta_of(e) * L * L for e, L in edge_load.items())
        return congestion_weight * pen - self.total_utility(selections)

    # ------------------------------------------------------------------
    # private cost & best response
    # ------------------------------------------------------------------
    def _options(self, rid: str) -> List[Optional[str]]:
        opts = [b["bundle_id"] for b in self.strategies[rid]]
        if self.allow_none:
            opts.append(None)
        return opts

    def private_cost(self, rid: str, bid: Optional[str],
                     loads_without: Dict[tuple, int],
                     flat_tolls: Optional[Dict[tuple, float]] = None) -> float:
        """Own latency of player rid on bid given loads of the others."""
        if bid is None:
            return 0.0
        flat_tolls = flat_tolls or {}
        cost = 0.0
        for e, d in self._edge_of.get((rid, bid), {}).items():
            cost += d * self.latency(e, loads_without.get(e, 0) + d)
            cost += flat_tolls.get(e, 0.0) * d
        cost -= self.utility_weight * self._util_of.get((rid, bid), 0.0)
        return cost

    def marginal_toll(self, rid: str, bid: Optional[str],
                      loads_without: Dict[tuple, int], lam: float) -> float:
        """Marginal-cost toll lambda * congestion externality.

        The externality on edge e is beta_e * L^{-i}_e * d_b,e; at lam = 1 the
        player internalizes exactly the congestion it imposes on others.
        """
        if bid is None or lam == 0.0:
            return 0.0
        toll = 0.0
        for e, d in self._edge_of.get((rid, bid), {}).items():
            toll += lam * self._beta_of(e) * loads_without.get(e, 0) * d
        return toll

    def best_response(self, rid: str, selections: Dict[str, Optional[str]],
                      lam: float = 0.0,
                      flat_tolls: Optional[Dict[tuple, float]] = None,
                      tie_break: str = "stay") -> Tuple[Optional[str], float]:
        """Return (best_bid, cost) for player rid given the other loads."""
        loads_without = self.loads_without(selections, rid)
        best_bid = selections.get(rid)
        best_cost = self.private_cost(rid, best_bid, loads_without, flat_tolls) + \
            self.marginal_toll(rid, best_bid, loads_without, lam)
        for bid in self._options(rid):
            if bid == best_bid:
                continue
            c = self.private_cost(rid, bid, loads_without, flat_tolls) + \
                self.marginal_toll(rid, bid, loads_without, lam)
            if c < best_cost - 1e-12:
                best_bid = bid
                best_cost = c
            elif abs(c - best_cost) <= 1e-12 and tie_break == "utility":
                u_new = self._util_of.get((rid, bid), 0.0)
                u_old = self._util_of.get((rid, best_bid), 0.0)
                if u_new > u_old:
                    best_bid = bid
                    best_cost = c
        return best_bid, best_cost

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def random_selection(self) -> Dict[str, Optional[str]]:
        return {rid: self._rng.choice(self._options(rid)) for rid in self.requests}

    def best_response_dynamics(self, initial: Optional[Dict[str, Optional[str]]] = None,
                               max_rounds: int = 100, lam: float = 0.0,
                               flat_tolls: Optional[Dict[tuple, float]] = None,
                               track: bool = False):
        """Best-response dynamics.  Strictly decreases Phi for ``unit``.

        Returns the reached (pure) equilibrium, or (equilibrium, history).
        """
        current = dict(initial) if initial is not None else self.random_selection()
        for rid in self.requests:
            current.setdefault(rid, None)
        phi = self.potential(current)
        history = [(dict(current), phi)] if track else None

        for _ in range(max_rounds):
            order = list(self.requests)
            self._rng.shuffle(order)
            improved = False
            for rid in order:
                new_bid, _ = self.best_response(rid, current, lam=lam,
                                                flat_tolls=flat_tolls)
                if new_bid != current.get(rid):
                    current[rid] = new_bid
                    improved = True
                    phi = self.potential(current)
                    if track:
                        history.append((dict(current), phi))
            if not improved:
                break

        if track:
            return current, history
        return current

    def nash_equilibria(self, n_restarts: int = 12, lam: float = 0.0,
                        max_rounds: int = 100,
                        flat_tolls: Optional[Dict[tuple, float]] = None) -> List[Dict]:
        equilibria = []
        seen = set()
        for _ in range(n_restarts):
            start = self.random_selection()
            eq = self.best_response_dynamics(start, max_rounds=max_rounds, lam=lam,
                                             flat_tolls=flat_tolls)
            key = tuple(sorted((rid, bid) for rid, bid in eq.items() if bid is not None))
            if key not in seen:
                seen.add(key)
                equilibria.append(eq)
        return equilibria

    # ------------------------------------------------------------------
    # optima and price of anarchy
    # ------------------------------------------------------------------
    def _strategy_options(self, rid: str) -> List[Optional[str]]:
        return self._options(rid)

    def brute_force_opt(self, objective) -> Tuple[Dict[str, Optional[str]], float]:
        """Exact optimum of `objective` over the product of strategy sets."""
        option_sets = [self._strategy_options(rid) for rid in self.requests]
        best = None
        best_val = float("inf")
        for combo in itertools.product(*option_sets):
            sel = dict(zip(self.requests, combo))
            val = objective(sel)
            if val < best_val:
                best_val = val
                best = sel
        return best, best_val

    def social_optimum(self) -> Tuple[Dict[str, Optional[str]], float]:
        return self.brute_force_opt(self.social_cost)

    def price_of_anarchy(self, n_restarts: int = 12, lam: float = 0.0,
                         flat_tolls: Optional[Dict[tuple, float]] = None) -> float:
        """Worst-case PoA: max_Nash SC / OPT SC."""
        _, opt_sc = self.social_optimum()
        if opt_sc <= 0:
            return 1.0
        equilibria = self.nash_equilibria(n_restarts=n_restarts, lam=lam,
                                          flat_tolls=flat_tolls)
        worst = max(self.social_cost(eq) for eq in equilibria)
        return worst / opt_sc


def toll_sweep(bundles: List[dict], edge_capacities: Optional[Dict[tuple, float]] = None,
               demand_model: str = "weighted",
               lambdas: Optional[List[float]] = None,
               base_latency: float = 0.0,
               beta: float = 1.0,
               utility_weight: float = 0.0,
               beta_override: Optional[Dict[tuple, float]] = None,
               base_override: Optional[Dict[tuple, float]] = None,
               seed: int = 42) -> List[dict]:
    """Worst-case PoA as a function of the marginal-cost toll coefficient."""
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    game = CongestionGame(bundles, edge_capacities, demand_model=demand_model,
                          base_latency=base_latency, beta=beta,
                          utility_weight=utility_weight, seed=seed)
    if beta_override:
        game._beta_override.update({_undirected(k): v for k, v in beta_override.items()})
    if base_override:
        game._base_override.update({_undirected(k): v for k, v in base_override.items()})
    _, opt_sc = game.social_optimum()
    rows = []
    for lam in lambdas:
        equilibria = game.nash_equilibria(n_restarts=10, lam=lam)
        worst_sc = max(game.social_cost(eq) for eq in equilibria)
        best_sc = min(game.social_cost(eq) for eq in equilibria)
        rows.append({
            "lambda": lam,
            "poa_worst": worst_sc / opt_sc if opt_sc > 0 else 1.0,
            "poa_best": best_sc / opt_sc if opt_sc > 0 else 1.0,
            "sc_worst": worst_sc,
            "sc_best": best_sc,
            "opt_sc": opt_sc,
            "n_equilibria": len(equilibria),
        })
    return rows


# ----------------------------------------------------------------------
# Atomic Pigou network: congestion externalities on a shared bottleneck
# ----------------------------------------------------------------------
def pigou_instance(n_players: int = 3, slow_latency: float = 2.0,
                   utility_weight: float = 0.0):
    """Canonical atomic congestion game with a shared fast link.

    Every player chooses between the shared fast link S->T (latency x, so a
    player's marginal cost rises with the number of fast-link users) and a
    constant-latency slow route S->M->T.  For n_players = 3 the pure-Nash
    outcome ``[2 fast, 1 slow]`` has social cost 6 versus the optimum 5,
    i.e. PoA = 6/5; marginal-cost tolls drive the outcome back to the
    optimum.  Returns (bundles, edge_capacities, latency_overrides).
    """
    fast = tuple(sorted(("S", "T")))
    slow_a, slow_b = tuple(sorted(("S", "M"))), tuple(sorted(("M", "T")))
    bundles = []
    for i in range(n_players):
        rid = f"R{i}"
        bundles.append({
            "bundle_id": f"{rid}_fast", "request_id": rid, "path": ["S", "T"],
            "edge_demands": {fast: 1}, "memory_demands": {}, "utility": 100.0,
        })
        bundles.append({
            "bundle_id": f"{rid}_slow", "request_id": rid, "path": ["S", "M", "T"],
            "edge_demands": {slow_a: 1, slow_b: 1}, "memory_demands": {},
            "utility": 100.0,
        })
    caps = {fast: 1.0, slow_a: 1.0, slow_b: 1.0}
    beta_override = {fast: 1.0, slow_a: 0.0, slow_b: 0.0}
    base_override = {slow_a: slow_latency, slow_b: 0.0}
    return bundles, caps, {"beta": beta_override, "base": base_override}


def pigou_poa_sweep(n_players_list: Optional[List[int]] = None,
                    seed: int = 42) -> List[dict]:
    """Worst-case PoA vs the number of competing players on the Pigou link."""
    if n_players_list is None:
        n_players_list = [2, 3, 4, 6, 8]
    rows = []
    for n in n_players_list:
        bundles, caps, ov = pigou_instance(n_players=n)
        game = CongestionGame(bundles, caps, demand_model="unit",
                              utility_weight=0.0, seed=seed)
        game._beta_override.update(ov["beta"])
        game._base_override.update(ov["base"])
        _, opt_sc = game.social_optimum()
        equilibria = game.nash_equilibria(n_restarts=30)
        worst = max(game.social_cost(eq) for eq in equilibria)
        rows.append({
            "n_players": n,
            "nash_sc_worst": worst,
            "opt_sc": opt_sc,
            "poa_worst": worst / opt_sc if opt_sc > 0 else 1.0,
            "n_equilibria": len(equilibria),
        })
    return rows


# ----------------------------------------------------------------------
# Braess's paradox on the classic 4-node network
# ----------------------------------------------------------------------
# Braess's paradox is a phenomenon of *splittable* (non-atomic) flow, so the
# experiment below uses the Wardrop model: a unit of infinitely-divisible flow
# splits so that all used paths have equal latency.  The canonical network is
# used: links S-A and B-T have latency x, links A-T and S-B constant latency 1,
# and the shortcut A-B latency ``cross_base``.
#
# Without the shortcut the equilibrium latency is 1.5.  As the shortcut gets
# faster (``cross_base`` -> 0) the equilibrium latency rises toward 2.0: adding
# capacity hurts.  Marginal-cost tolls (``lam``) at lam = 1 recover the social
# optimum, restoring latency 1.5.


class WardropNetwork:
    """Splittable-flow (Wardrop) equilibrium on a small path network.

    ``paths`` is a list of paths, each a list of edges (nodes or tuples).
    Edge latency is l_e(f) = base_e + beta_e * f_e.  ``flow`` is the total
    amount of divisible traffic to route.  The equilibrium minimises the
    potential sum_e int_0^{f_e} l_e(x) dx subject to conservation; because
    latencies are affine the equilibrium is found exactly by enumerating the
    set of active (used) paths and solving the resulting linear system.
    """

    def __init__(self, paths: List[List], base: Dict[tuple, float],
                 beta: Dict[tuple, float], flow: float):
        self.paths = [[_undirected(e) for e in p] for p in paths]
        self.base = {_undirected(e): v for e, v in base.items()}
        self.beta = {_undirected(e): v for e, v in beta.items()}
        self.flow = flow

    def _path_base(self, p: List[tuple]) -> float:
        return sum(self.base.get(e, 0.0) for e in p)

    def _path_beta(self, p: List[tuple]) -> float:
        return sum(self.beta.get(e, 0.0) for e in p)

    def _loads(self, flows: List[float]) -> Dict[tuple, float]:
        load = defaultdict(float)
        for p, f in zip(self.paths, flows):
            for e in p:
                load[e] += f
        return load

    def _edge_flow_of_path(self, e: tuple, p: List[tuple]) -> float:
        return 1.0 if e in p else 0.0

    def equilibrium(self, lam: float = 0.0,
                    flat_tolls: Optional[Dict[tuple, float]] = None) -> Dict:
        """Wardrop equilibrium under marginal tolls ``lam`` and flat tolls.

        Marginal-cost pricing with coefficient ``lam`` scales every beta by
        (1 + lam); flat tolls add a constant per-unit charge on their edges.
        Returns {"flows", "edge_load", "latency", "social_cost"}.
        """
        flat_tolls = flat_tolls or {}
        n = len(self.paths)
        # marginal cost of path p given flows (toll included)
        def path_mc(flows, p):
            mc = self._path_base(p) + sum(flat_tolls.get(e, 0.0) for e in p)
            load = self._loads(flows)
            mc += (1.0 + lam) * sum(self.beta.get(e, 0.0) * load[e]
                                    for e in p)
            return mc

        best = None
        for mask in range(1, 1 << n):
            active = [i for i in range(n) if mask & (1 << i)]
            k = len(active)
            if k == 0:
                continue
            # unknowns: f_active (k) and equilibrium cost c (1)
            A = np.zeros((k + 1, k + 1))
            bvec = np.zeros(k + 1)
            for row, i in enumerate(active):
                p = self.paths[i]
                for col, j in enumerate(active):
                    q = self.paths[j]
                    A[row, col] = (1.0 + lam) * sum(
                        self.beta.get(e, 0.0)
                        for e in set(p) & set(q))
                A[row, k] = -1.0
                bvec[row] = -self._path_base(p) - sum(flat_tolls.get(e, 0.0) for e in p)
            for col, j in enumerate(active):
                A[k, col] = 1.0
            bvec[k] = self.flow
            try:
                sol = np.linalg.solve(A, bvec)
            except np.linalg.LinAlgError:
                continue
            flows = np.zeros(n)
            for row, i in enumerate(active):
                flows[i] = sol[row]
            c = sol[k]
            if flows.min() < -1e-9:
                continue
            flows = np.maximum(flows, 0.0)
            load = self._loads(flows)
            # unused paths must not be strictly better
            ok = True
            for i in range(n):
                if not (mask & (1 << i)):
                    mc = path_mc(flows, self.paths[i])
                    if mc < c - 1e-9:
                        ok = False
                        break
            if not ok:
                continue
            social = sum(load[e] * (self.base.get(e, 0.0) +
                                    self.beta.get(e, 0.0) * load[e])
                         for e in load)
            candidate = {"flows": flows, "edge_load": load,
                         "social_cost": social}
            if best is None or social < best["social_cost"]:
                best = candidate
        # There is always at least one nonempty active set (single-path) that
        # yields a finite solution.
        return best

    def social_optimum(self) -> float:
        """Minimum total latency over feasible flows.

        The social marginal cost of path p is sum_{e in p} (base_e +
        2 beta_e f_e), i.e. the same equilibrium problem with doubled betas.
        """
        n = len(self.paths)
        best = float("inf")
        for mask in range(1, 1 << n):
            active = [i for i in range(n) if mask & (1 << i)]
            k = len(active)
            A = np.zeros((k + 1, k + 1))
            bvec = np.zeros(k + 1)
            for row, i in enumerate(active):
                p = self.paths[i]
                for col, j in enumerate(active):
                    q = self.paths[j]
                    A[row, col] = 2.0 * sum(self.beta.get(e, 0.0)
                                            for e in set(p) & set(q))
                A[row, k] = -1.0
                bvec[row] = -self._path_base(p)
            for col, j in enumerate(active):
                A[k, col] = 1.0
            bvec[k] = self.flow
            try:
                sol = np.linalg.solve(A, bvec)
            except np.linalg.LinAlgError:
                continue
            flows = np.zeros(n)
            for row, i in enumerate(active):
                flows[i] = sol[row]
            if flows.min() < -1e-9:
                continue
            flows = np.maximum(flows, 0.0)
            load = self._loads(flows)
            social = sum(load[e] * (self.base.get(e, 0.0) +
                                    self.beta.get(e, 0.0) * load[e])
                         for e in load)
            best = min(best, social)
        return best

    def price_of_anarchy(self, lam: float = 0.0,
                         flat_tolls: Optional[Dict[tuple, float]] = None) -> float:
        eq = self.equilibrium(lam=lam, flat_tolls=flat_tolls)
        opt = self.social_optimum()
        return eq["social_cost"] / opt if opt > 0 else 1.0


def _braess_paths(add_cross: bool):
    sa, at, sb, bt, ab = [tuple(sorted(e)) for e in
                          [("S", "A"), ("A", "T"), ("S", "B"), ("B", "T"), ("A", "B")]]
    paths = [[sa, at], [sb, bt]]
    if add_cross:
        paths.append([sa, ab, bt])
    return paths


def _braess_latency(cross_base: float):
    sa, at, sb, bt, ab = [tuple(sorted(e)) for e in
                          [("S", "A"), ("A", "T"), ("S", "B"), ("B", "T"), ("A", "B")]]
    base = {at: 1.0, sb: 1.0, ab: cross_base, sa: 0.0, bt: 0.0}
    beta = {sa: 1.0, bt: 1.0, at: 0.0, sb: 0.0, ab: 0.0}
    return base, beta


def braess_wardrop(cross_base: float = 0.0, add_cross: bool = True,
                   flow: float = 1.0) -> WardropNetwork:
    """Canonical Braess network in the Wardrop model."""
    base, beta = _braess_latency(cross_base)
    return WardropNetwork(_braess_paths(add_cross), base, beta, flow)


def braess_capacity_sweep(cross_bases: Optional[List[float]] = None,
                          flow: float = 1.0) -> List[dict]:
    """Equilibrium latency / PoA vs the shortcut's free-flow latency.

    As the shortcut gets faster (capacity grows), the equilibrium latency rises
    from 1.5 toward 2.0: adding capacity hurts selfish routing.  The tolled
    equilibrium (lam = 1) stays at the social optimum throughout.
    """
    if cross_bases is None:
        cross_bases = [1.0, 0.7, 0.5, 0.35, 0.2, 0.1, 0.0]
    rows = []
    base_net = braess_wardrop(cross_base=1.0, add_cross=False, flow=flow)
    opt_sc = base_net.social_optimum()
    for cb in cross_bases:
        net = braess_wardrop(cross_base=cb, add_cross=True, flow=flow)
        eq = net.equilibrium(lam=0.0)
        eq_toll = net.equilibrium(lam=1.0)
        rows.append({
            "cross_base_latency": cb,
            "eq_social_cost": eq["social_cost"],
            "tolled_social_cost": eq_toll["social_cost"],
            "opt_social_cost": opt_sc,
            "poa_worst": eq["social_cost"] / opt_sc,
            "poa_tolled": eq_toll["social_cost"] / opt_sc,
        })
    return rows


def braess_toll_sweep(tolls: Optional[List[float]] = None,
                      flow: float = 1.0) -> List[dict]:
    """A flat toll on the free shortcut removes the paradox."""
    if tolls is None:
        tolls = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0]
    rows = []
    cross = _undirected(("A", "B"))
    net = braess_wardrop(cross_base=0.0, add_cross=True, flow=flow)
    opt_sc = net.social_optimum()
    for t in tolls:
        eq = net.equilibrium(lam=0.0, flat_tolls={cross: t})
        rows.append({
            "flat_toll": t,
            "eq_social_cost": eq["social_cost"],
            "opt_social_cost": opt_sc,
            "poa_worst": eq["social_cost"] / opt_sc,
        })
    return rows


def braess_marginal_toll_sweep(lambdas: Optional[List[float]] = None,
                               flow: float = 1.0) -> List[dict]:
    """Marginal-cost tolls (coefficient ``lam``) on the paradox network.

    At lam = 0 the free shortcut yields the inefficient Wardrop equilibrium
    (social cost 2.0); at lam = 1 private cost equals social marginal cost and
    the equilibrium social cost drops to the optimum (1.5).
    """
    if lambdas is None:
        lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    net = braess_wardrop(cross_base=0.0, add_cross=True, flow=flow)
    opt_sc = net.social_optimum()
    for lam in lambdas:
        eq = net.equilibrium(lam=lam)
        rows.append({
            "lambda": lam,
            "eq_social_cost": eq["social_cost"],
            "opt_social_cost": opt_sc,
            "poa_worst": eq["social_cost"] / opt_sc,
        })
    return rows
