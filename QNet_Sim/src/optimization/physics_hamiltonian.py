from collections import defaultdict
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple


class HamiltonianTerm:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def energy(self, selections: Dict[Tuple[str, str], int],
               bundles: List[dict], edge_capacities: dict,
               memory_capacities: dict) -> float:
        raise NotImplementedError

    def label(self) -> str:
        return self.name


class UtilityTerm(HamiltonianTerm):
    def __init__(self, weight: float = 1.0):
        super().__init__("utility", weight)

    def energy(self, selections, bundles, edge_capacities, memory_capacities):
        total = 0.0
        for b in bundles:
            key = (b["request_id"], b["bundle_id"])
            total -= b["utility"] * selections.get(key, 0)
        return self.weight * total


class OnePerRequestTerm(HamiltonianTerm):
    def __init__(self, weight: float = 1.0):
        super().__init__("one_per_request", weight)

    def energy(self, selections, bundles, edge_capacities, memory_capacities):
        total = 0.0
        req_bundles = defaultdict(list)
        for b in bundles:
            req_bundles[b["request_id"]].append(b)
        for rid, bs in req_bundles.items():
            if len(bs) < 2:
                continue
            active = 0
            for b in bs:
                key = (b["request_id"], b["bundle_id"])
                active += selections.get(key, 0)
            if active > 1:
                total += active * (active - 1) / 2
        return self.weight * total


class EdgeCapacityTerm(HamiltonianTerm):
    def __init__(self, weight: float = 1.0, exponent: float = 2.0):
        super().__init__("edge_capacity", weight)
        self.exponent = exponent

    def energy(self, selections, bundles, edge_capacities, memory_capacities):
        total = 0.0
        edge_load = defaultdict(int)
        for b in bundles:
            key = (b["request_id"], b["bundle_id"])
            if not selections.get(key, 0):
                continue
            for edge, d in b["edge_demands"].items():
                edge_load[tuple(sorted(edge))] += d
        for edge, load in edge_load.items():
            cap = edge_capacities.get(edge, 0)
            if load > cap:
                total += (load - cap) ** self.exponent
        return self.weight * total


class MemoryCapacityTerm(HamiltonianTerm):
    def __init__(self, weight: float = 1.0, exponent: float = 2.0):
        super().__init__("memory_capacity", weight)
        self.exponent = exponent

    def energy(self, selections, bundles, edge_capacities, memory_capacities):
        total = 0.0
        mem_load = defaultdict(int)
        for b in bundles:
            key = (b["request_id"], b["bundle_id"])
            if not selections.get(key, 0):
                continue
            for node, d in b["memory_demands"].items():
                mem_load[node] += d
        for node, load in mem_load.items():
            cap = memory_capacities.get(node, 0)
            if load > cap:
                total += (load - cap) ** self.exponent
        return self.weight * total


class SoftCongestionTerm(HamiltonianTerm):
    def __init__(self, weight: float = 1.0, warning_threshold: float = 0.7,
                 exponent: float = 2.0):
        super().__init__("soft_congestion", weight)
        self.warning_threshold = warning_threshold
        self.exponent = exponent

    def energy(self, selections, bundles, edge_capacities, memory_capacities):
        total = 0.0
        edge_load = defaultdict(int)
        for b in bundles:
            key = (b["request_id"], b["bundle_id"])
            if not selections.get(key, 0):
                continue
            for edge, d in b["edge_demands"].items():
                edge_load[tuple(sorted(edge))] += d
        for edge, load in edge_load.items():
            cap = edge_capacities.get(edge, 1)
            ratio = load / cap
            if ratio > self.warning_threshold:
                overshoot = ratio - self.warning_threshold
                total += overshoot ** self.exponent
        return self.weight * total

    def label(self) -> str:
        return f"congestion({self.warning_threshold})"


class MemoryRiskTerm(HamiltonianTerm):
    def __init__(self, weight: float = 1.0, tau_mem: float = 1.0):
        super().__init__("memory_risk", weight)
        self.tau_mem = tau_mem

    def energy(self, selections, bundles, edge_capacities, memory_capacities):
        import math
        total = 0.0
        for b in bundles:
            key = (b["request_id"], b["bundle_id"])
            if not selections.get(key, 0):
                continue
            wait_time = b.get("latency", 0.0)
            risk = 1.0 - math.exp(-wait_time / self.tau_mem)
            total += risk
        return self.weight * total

    def label(self) -> str:
        return f"mem_risk(tau={self.tau_mem})"


class PhysicalHamiltonian:
    def __init__(self, bundles: List[dict],
                 edge_capacities: Dict[Tuple[str, str], int],
                 memory_capacities: Dict[str, int]):
        self.bundles = bundles
        self.edge_capacities = {tuple(sorted(k)): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self.terms: List[HamiltonianTerm] = []
        self._default()

    def _default(self):
        self.terms = [
            UtilityTerm(weight=1.0),
            OnePerRequestTerm(weight=100.0),
            EdgeCapacityTerm(weight=10.0),
            MemoryCapacityTerm(weight=10.0),
        ]

    def clear_terms(self):
        self.terms = []

    def add_term(self, term: HamiltonianTerm):
        self.terms.append(term)

    def set_terms(self, terms: List[HamiltonianTerm]):
        self.terms = list(terms)

    def energy(self, selections: Dict[Tuple[str, str], int]) -> float:
        total = 0.0
        for term in self.terms:
            total += term.energy(selections, self.bundles,
                                 self.edge_capacities, self.memory_capacities)
        return total

    def decode(self, variable_assignment: Dict[str, int]) -> List[Tuple[str, str]]:
        selected = []
        for b in self.bundles:
            key = (b["request_id"], b["bundle_id"])
            var_name = f"x_{self.bundles.index(b)}"
            if variable_assignment.get(var_name, 0) == 1:
                selected.append(key)
        return selected

    def evaluate_from_variables(self, variable_assignment: Dict[str, int]) -> float:
        selections = {}
        for i, b in enumerate(self.bundles):
            key = (b["request_id"], b["bundle_id"])
            var_name = f"x_{i}"
            selections[key] = variable_assignment.get(var_name, 0)
        return self.energy(selections)

    def to_qubo_slackfree(self, utility_weight=1.0,
                          one_per_request_weight=100.0,
                          congestion_weight=10.0,
                          memory_ratio_weight=10.0,
                          exponent=2.0) -> Tuple[Dict[Tuple[str, str], float], float]:
        """Extension 12.1: direct (slack-free) effective Hamiltonian.

        H = -sum_rb u_{r,b} x_{r,b}
          + lambda_req sum_r (sum_b x_{r,b} - 1)^2
          + lambda_cong sum_e (L_e / C_e)^p
          + lambda_mem sum_n (M_n / M_n^max)^p

        with rho(z)=z^p a smooth load-to-capacity ratio penalty.  The only
        variables are the bundle binaries: no LogEncInteger slack variables,
        so the QUBO has exactly |B| binaries (vs |B| + slack for Eq. 5.3-5.4).
        """
        from pyqubo import Binary

        variables = {}
        for i, b in enumerate(self.bundles):
            key = (b["request_id"], b["bundle_id"])
            variables[key] = Binary(f"x_{i}")

        hamiltonian = 0
        for b in self.bundles:
            key = (b["request_id"], b["bundle_id"])
            hamiltonian -= utility_weight * b["utility"] * variables[key]

        req_bundles = defaultdict(list)
        for b in self.bundles:
            req_bundles[b["request_id"]].append(b)
        for rid, bs in req_bundles.items():
            if len(bs) < 2:
                continue
            load_expr = sum(variables[(b["request_id"], b["bundle_id"])] for b in bs)
            hamiltonian += one_per_request_weight * (load_expr - 1) ** 2

        for edge, cap in self.edge_capacities.items():
            if cap <= 0:
                continue
            load_expr = sum(
                b["edge_demands"].get(edge, 0) * variables[(b["request_id"], b["bundle_id"])]
                for b in self.bundles
            )
            hamiltonian += congestion_weight * (load_expr / cap) ** int(exponent)

        for node, cap in self.memory_capacities.items():
            if cap <= 0:
                continue
            load_expr = sum(
                b["memory_demands"].get(node, 0) * variables[(b["request_id"], b["bundle_id"])]
                for b in self.bundles
            )
            hamiltonian += memory_ratio_weight * (load_expr / cap) ** int(exponent)

        return hamiltonian.compile().to_qubo()

    def to_qubo(self, **term_weights) -> Tuple[Dict[Tuple[str, str], float], float]:
        from pyqubo import Binary, Constraint, LogEncInteger

        variables = {}
        for i, b in enumerate(self.bundles):
            key = (b["request_id"], b["bundle_id"])
            variables[key] = Binary(f"x_{i}")

        A = term_weights.get("A", 100.0)
        B = term_weights.get("B", 10.0)
        D = term_weights.get("D", 10.0)
        cong_weight = term_weights.get("congestion", 0.0)
        cong_threshold = term_weights.get("congestion_threshold", 0.7)
        risk_weight = term_weights.get("risk", 0.0)
        risk_tau = term_weights.get("risk_tau", 1.0)

        hamiltonian = 0
        for key, var in variables.items():
            for b in self.bundles:
                bk = (b["request_id"], b["bundle_id"])
                if bk == key:
                    hamiltonian -= b["utility"] * var

        req_bundles = defaultdict(list)
        for b in self.bundles:
            req_bundles[b["request_id"]].append(b)
        for rid, bs in req_bundles.items():
            if len(bs) < 2:
                continue
            conflict = 0
            for first, second in combinations(bs, 2):
                fk = (first["request_id"], first["bundle_id"])
                sk = (second["request_id"], second["bundle_id"])
                conflict += variables[fk] * variables[sk]
            hamiltonian += A * Constraint(conflict, label=f"req_{rid}")

        for e_idx, (edge, cap) in enumerate(self.edge_capacities.items()):
            load = 0
            for b in self.bundles:
                key = (b["request_id"], b["bundle_id"])
                if edge in b.get("edge_demands", {}):
                    load += b["edge_demands"][edge] * variables[key]
            slack = LogEncInteger(f"edge_{e_idx}_slack", (0, cap)) if cap > 0 else 0
            hamiltonian += B * Constraint((load + slack - cap) ** 2, label=f"edge_{e_idx}")

        for m_idx, (node, cap) in enumerate(self.memory_capacities.items()):
            load = 0
            for b in self.bundles:
                key = (b["request_id"], b["bundle_id"])
                if node in b.get("memory_demands", {}):
                    load += b["memory_demands"][node] * variables[key]
            slack = LogEncInteger(f"mem_{m_idx}_slack", (0, cap)) if cap > 0 else 0
            hamiltonian += D * Constraint((load + slack - cap) ** 2, label=f"mem_{m_idx}")

        if cong_weight > 0:
            for e_idx, (edge, cap) in enumerate(self.edge_capacities.items()):
                load = 0
                for b in self.bundles:
                    key = (b["request_id"], b["bundle_id"])
                    if edge in b.get("edge_demands", {}):
                        load += b["edge_demands"][edge] * variables[key]
                hamiltonian += cong_weight * Constraint(
                    (load / max(cap, 1) - cong_threshold) ** 2 * (load > cong_threshold * cap),
                    label=f"cong_{e_idx}"
                )

        if risk_weight > 0:
            import math
            risk_sum = 0
            for b in self.bundles:
                key = (b["request_id"], b["bundle_id"])
                wait = b.get("latency", 0.0)
                risk_val = 1.0 - math.exp(-wait / risk_tau)
                risk_sum += risk_val * variables[key]
            hamiltonian += risk_weight * risk_sum

        if isinstance(hamiltonian, (int, float)):
            from pyqubo import Num
            return Num(hamiltonian).compile().to_qubo()
        return hamiltonian.compile().to_qubo()

    def to_bqm(self, **term_weights):
        Q, offset = self.to_qubo(**term_weights)
        import dimod
        return dimod.BQM.from_qubo(Q, offset)

    def describe(self) -> str:
        parts = []
        for t in self.terms:
            parts.append(f"{t.label()} (w={t.weight})")
        return " + ".join(parts) if parts else "(empty)"
