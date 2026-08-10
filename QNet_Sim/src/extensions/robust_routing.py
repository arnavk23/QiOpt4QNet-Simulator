"""Extension D2: robust / stochastic routing under uncertain utilities.

Quantum-network bundles carry an estimated utility (delivered fidelity and
latency under the current noise model).  In practice that estimate is
uncertain: link fidelities drift, purifications fail, and requests arrive
unexpectedly.  This module routes against a set of utility *scenarios* and
solves either the nominal problem (expected utilities) or a Gamma-robust
problem whose per-bundle coefficient is

    u_b(gamma) = u_b_nominal - gamma * (u_b_nominal - min_s u_b(s)).

gamma = 0 is the nominal (expected-utility) router; gamma = 1 is the maximin
(worst-case) router.  Robust solutions sacrifice a little nominal utility to
raise the worst-case outcome and cut worst-case regret.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from optimization.metropolis_annealer import MetropolisAnnealer


def bundle_key(bundle: dict) -> Tuple[str, str]:
    return (bundle["request_id"], bundle["bundle_id"])


def generate_scenarios(bundles: List[dict], n_scenarios: int = 10,
                       noise_scale: float = 0.15, failure_prob: float = 0.25,
                       severity: float = 0.5, seed: int = 42,
                       include_nominal: bool = True) -> List[Dict[tuple, float]]:
    """Sample utility scenarios from a correlated noise + link-failure model.

    Each scenario starts from the nominal bundle utilities and applies (a)
    independent log-normal measurement noise and (b), with probability
    ``failure_prob``, a structured event that degrades every bundle routed
    through one random edge by a factor ``(1 - severity)``.  ``include_nominal``
    prepends the noise-free scenario so the set is never empty.
    """
    rng = random.Random(seed)
    keys = [bundle_key(b) for b in bundles]
    nominal = {k: b["utility"] for b, k in zip(bundles, keys)}
    edges = sorted({e for b in bundles for e in b["edge_demands"]})

    scenarios: List[Dict[tuple, float]] = []
    if include_nominal:
        scenarios.append(dict(nominal))
    for _ in range(max(0, n_scenarios - (1 if include_nominal else 0))):
        failed_edge = rng.choice(edges) if rng.random() < failure_prob else None
        scenario = {}
        for b, k in zip(bundles, keys):
            u = nominal[k] * max(0.0, 1.0 + rng.gauss(0.0, noise_scale))
            if failed_edge is not None and failed_edge in b["edge_demands"]:
                u *= (1.0 - severity)
            scenario[k] = u
        scenarios.append(scenario)
    return scenarios


class RobustRoutingModel:
    def __init__(self, bundles: List[dict], edge_capacities: Dict[tuple, float],
                 memory_capacities: Dict[str, float], scenarios: List[Dict[tuple, float]],
                 seed: int = 42):
        self.bundles = bundles
        self.edge_capacities = edge_capacities
        self.memory_capacities = memory_capacities
        self.seed = seed
        self.scenarios = scenarios
        self.keys = [bundle_key(b) for b in bundles]
        self.nominal = {bundle_key(b): b["utility"] for b in bundles}
        if scenarios:
            self._min = {k: min(s.get(k, self.nominal[k]) for s in scenarios)
                         for k in self.keys}
        else:
            self._min = dict(self.nominal)
        self._hindsight_cache: Dict[Tuple, Dict[str, Optional[str]]] = {}

    # ------------------------------------------------------------------
    # coefficients & objectives
    # ------------------------------------------------------------------
    def robust_coefficient(self, key: Tuple[str, str], gamma: float) -> float:
        return self.nominal[key] - gamma * (self.nominal[key] - self._min[key])

    def objective(self, selections: Dict[str, Optional[str]], mode: str = "nominal",
                  gamma: float = 0.0) -> float:
        """Utility of ``selections`` under a chosen objective.

        ``mode`` is one of "nominal", "worst", "mean", or "robust" (Gamma-robust
        with coefficient ``gamma``).
        """
        if mode == "nominal":
            weights = self.nominal
        elif mode == "worst":
            weights = self._min
        elif mode == "mean":
            weights = {k: sum(s.get(k, self.nominal[k]) for s in self.scenarios)
                       / max(len(self.scenarios), 1) for k in self.keys}
        elif mode == "robust":
            weights = {k: self.robust_coefficient(k, gamma) for k in self.keys}
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return sum(weights.get((rid, bid), 0.0)
                   for rid, bid in selections.items() if bid is not None)

    def scenario_values(self, selections: Dict[str, Optional[str]]) -> List[float]:
        return [sum(s.get((rid, bid), 0.0) for rid, bid in selections.items()
                    if bid is not None) for s in self.scenarios]

    # ------------------------------------------------------------------
    # solving
    # ------------------------------------------------------------------
    def _solve_utilities(self, utility_of: Dict[Tuple[str, str], float],
                         **solver_kwargs) -> Dict[str, Optional[str]]:
        clones = []
        for b in self.bundles:
            cb = dict(b)
            cb["utility"] = utility_of[bundle_key(b)]
            clones.append(cb)
        opt = MetropolisAnnealer(clones, self.edge_capacities, self.memory_capacities,
                                 seed=self.seed)
        result = opt.solve(**solver_kwargs)
        selections = {rid: None for rid in {k[0] for k in self.keys}}
        for rid, bid in result.get("selected", []):
            selections[rid] = bid
        return selections

    def solve(self, gamma: float = 0.0, **solver_kwargs) -> Dict[str, Optional[str]]:
        """Solve the Gamma-robust problem (gamma=0: nominal; gamma=1: maximin)."""
        coef = {k: self.robust_coefficient(k, gamma) for k in self.keys}
        return self._solve_utilities(coef, **solver_kwargs)

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def evaluate(self, selections: Dict[str, Optional[str]]) -> dict:
        vals = self.scenario_values(selections)
        worst = min(vals) if vals else 0.0
        mean = sum(vals) / max(len(vals), 1) if vals else 0.0
        std = (sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)) ** 0.5 if vals else 0.0
        return {
            "nominal_util": self.objective(selections, "nominal"),
            "worst_util": worst,
            "mean_util": mean,
            "std_util": std,
            "served": sum(1 for bid in selections.values() if bid is not None),
            "n_requests": len(selections),
        }

    def _best_in_hindsight(self, scenario: Dict[tuple, float],
                           **solver_kwargs) -> Dict[str, Optional[str]]:
        cache_key = tuple(sorted((k, round(v, 6)) for k, v in scenario.items()))
        if cache_key not in self._hindsight_cache:
            self._hindsight_cache[cache_key] = self._solve_utilities(scenario, **solver_kwargs)
        return self._hindsight_cache[cache_key]

    # ------------------------------------------------------------------
    # feasibility & exact solving
    # ------------------------------------------------------------------
    def _feasible(self, selections: Dict[str, Optional[str]]) -> bool:
        edge_load: Dict[tuple, float] = defaultdict(float)
        mem_load: Dict[str, float] = defaultdict(float)
        for b in self.bundles:
            key = bundle_key(b)
            if selections.get(key[0]) != key[1]:
                continue
            for e, d in b["edge_demands"].items():
                edge_load[tuple(sorted(e))] += d
            for node, d in b["memory_demands"].items():
                mem_load[node] += d
        for e, load in edge_load.items():
            if load > self.edge_capacities.get(e, 1e18):
                return False
        for node, load in mem_load.items():
            if load > self.memory_capacities.get(node, 1e18):
                return False
        return True

    def solve_exact(self, gamma: float = 0.0,
                    max_combos: int = 20000) -> Dict[str, Optional[str]]:
        """Exact brute-force maximiser of the Gamma-robust objective.

        Only feasible for small instances (``product |strategies_i| <= max_combos``).
        """
        import itertools
        groups: Dict[str, List[dict]] = {}
        for b in self.bundles:
            groups.setdefault(b["request_id"], []).append(b)
        orders = [groups[r] for r in groups]
        n_combos = 1
        for bs in orders:
            n_combos *= len(bs)
        if n_combos > max_combos:
            raise ValueError(f"{n_combos} combinations exceeds max_combos={max_combos}")
        coef = {k: self.robust_coefficient(k, gamma) for k in self.keys}
        best_selection = None
        best_value = -float("inf")
        for combo in itertools.product(*orders):
            selections = {b["request_id"]: b["bundle_id"] for b in combo}
            if not self._feasible(selections):
                continue
            value = sum(coef[bundle_key(b)] for b in combo)
            if value > best_value:
                best_value = value
                best_selection = selections
        return best_selection

    def _exact_best_per_scenario(self, scenario: Dict[tuple, float]) -> float:
        clones = []
        for b in self.bundles:
            cb = dict(b)
            cb["utility"] = scenario[bundle_key(b)]
            clones.append(cb)
        model = RobustRoutingModel(clones, self.edge_capacities, self.memory_capacities,
                                   [scenario], seed=self.seed)
        best = model.solve_exact(0.0)
        return sum(scenario[bundle_key(b)] for b in self.bundles
                   if best.get(b["request_id"]) == b["bundle_id"])

    def solve_exact_regret(self, max_combos: int = 20000) -> Dict[str, Optional[str]]:
        """Exact min-max-regret routing (a separate robust criterion from maximin).

        The maximin solution (``solve_exact(1.0)``) maximises the worst-case
        utility; the min-max-regret solution minimises the worst gap to the
        best-in-hindsight outcome.  The two can differ.  Ties are broken by
        mean regret, then nominal utility.
        """
        import itertools
        groups: Dict[str, List[dict]] = {}
        for b in self.bundles:
            groups.setdefault(b["request_id"], []).append(b)
        orders = [groups[r] for r in groups]
        n_combos = 1
        for bs in orders:
            n_combos *= len(bs)
        if n_combos > max_combos:
            raise ValueError(f"{n_combos} combinations exceeds max_combos={max_combos}")
        hindsight = [self._exact_best_per_scenario(s) for s in self.scenarios]
        best_selection = None
        best_key = None
        for combo in itertools.product(*orders):
            selections = {b["request_id"]: b["bundle_id"] for b in combo}
            if not self._feasible(selections):
                continue
            regrets = []
            for s, h in zip(self.scenarios, hindsight):
                mine = sum(s.get((rid, bid), 0.0) for rid, bid in selections.items()
                           if bid is not None)
                regrets.append(max(0.0, h - mine))
            max_reg = max(regrets)
            mean_reg = sum(regrets) / max(len(regrets), 1)
            nominal = self.objective(selections, "nominal")
            key = (max_reg, mean_reg, -nominal)
            if best_key is None or key < best_key:
                best_key = key
                best_selection = selections
        return best_selection

    def regret(self, selections: Dict[str, Optional[str]],
               exact: bool = False, **solver_kwargs) -> dict:
        """Max / mean regret against the best-in-hindsight per scenario."""
        regrets = []
        for s in self.scenarios:
            if exact:
                clones = []
                for b in self.bundles:
                    cb = dict(b)
                    cb["utility"] = s[bundle_key(b)]
                    clones.append(cb)
                model = RobustRoutingModel(clones, self.edge_capacities,
                                           self.memory_capacities, [s], seed=self.seed)
                best = model.solve_exact(0.0)
            else:
                best = self._best_in_hindsight(s, **solver_kwargs)
            best_val = sum(s.get((rid, bid), 0.0) for rid, bid in best.items()
                           if bid is not None)
            mine = sum(s.get((rid, bid), 0.0) for rid, bid in selections.items()
                       if bid is not None)
            regrets.append(max(0.0, best_val - mine))
        return {"max_regret": max(regrets), "mean_regret": sum(regrets) / max(len(regrets), 1)}

    # ------------------------------------------------------------------
    # sweeps
    # ------------------------------------------------------------------
    def pareto_sweep(self, gammas: Optional[List[float]] = None,
                     exact: bool = False, **solver_kwargs) -> List[dict]:
        """Nominal vs worst-case utility trade-off across the robustness budget."""
        if gammas is None:
            gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        rows = []
        for gamma in gammas:
            if exact:
                selections = self.solve_exact(gamma)
            else:
                selections = self.solve(gamma, **solver_kwargs)
            ev = self.evaluate(selections)
            reg = self.regret(selections, exact=exact, **solver_kwargs)
            rows.append({
                "gamma": gamma,
                "nominal_util": ev["nominal_util"],
                "worst_util": ev["worst_util"],
                "mean_util": ev["mean_util"],
                "std_util": ev["std_util"],
                "max_regret": reg["max_regret"],
                "mean_regret": reg["mean_regret"],
                "served": ev["served"],
            })
        return rows

    def robustness_gain(self, exact: bool = False, **solver_kwargs) -> dict:
        """Worst-case utility / regret gain of maximin vs nominal routing."""
        if exact:
            nominal = self.solve_exact(0.0)
            robust = self.solve_exact(1.0)
        else:
            nominal = self.solve(0.0, **solver_kwargs)
            robust = self.solve(1.0, **solver_kwargs)
        ev_n = self.evaluate(nominal)
        ev_r = self.evaluate(robust)
        reg_n = self.regret(nominal, exact=exact, **solver_kwargs)
        reg_r = self.regret(robust, exact=exact, **solver_kwargs)
        return {
            "nominal_worst_util": ev_n["worst_util"],
            "robust_worst_util": ev_r["worst_util"],
            "worst_util_gain": ev_r["worst_util"] - ev_n["worst_util"],
            "nominal_nominal_util": ev_n["nominal_util"],
            "robust_nominal_util": ev_r["nominal_util"],
            "nominal_util_loss": ev_n["nominal_util"] - ev_r["nominal_util"],
            "nominal_max_regret": reg_n["max_regret"],
            "robust_max_regret": reg_r["max_regret"],
            "max_regret_gain": reg_n["max_regret"] - reg_r["max_regret"],
        }
