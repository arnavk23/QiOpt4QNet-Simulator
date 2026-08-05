import math
import random
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple


class StreamingAnnealer:
    def __init__(self, edge_capacities: dict, memory_capacities: dict,
                 seed: Optional[int] = None,
                 congestion_weight: float = 0.0,
                 congestion_threshold: float = 0.7,
                 risk_weight: float = 0.0, risk_tau: float = 1.0,
                 use_fidelity_risk: bool = False, hold_scale: float = 1.0):
        self.edge_capacities = {tuple(sorted(k)): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self._rng = random.Random(seed)
        self.congestion_weight = congestion_weight
        self.congestion_threshold = congestion_threshold
        self.risk_weight = risk_weight
        self.risk_tau = risk_tau
        self.use_fidelity_risk = use_fidelity_risk
        self.hold_scale = hold_scale

        self._bundles: Dict[str, List[dict]] = {}
        self._bundle_map: Dict[Tuple[str, str], dict] = {}
        self._util_of: Dict[Tuple[str, str], float] = {}
        self._edge_of: Dict[Tuple[str, str], dict] = {}
        self._mem_of: Dict[Tuple[str, str], dict] = {}
        self._latency_of: Dict[Tuple[str, str], float] = {}
        self._fidelity_of: Dict[Tuple[str, str], float] = {}

        self.selections: Dict[str, Optional[str]] = {}
        self._selections_history: List[Tuple[str, Optional[str]]] = []

        self._arrival_order: List[str] = []
        self._active_requests: set = set()

    def add_request(self, request_id: str, bundles: List[dict]):
        self._bundles[request_id] = bundles
        self._arrival_order.append(request_id)
        self._active_requests.add(request_id)

        if request_id not in self.selections:
            best = self._pick_best_feasible(request_id, bundles, self.selections)
            self.selections[request_id] = best
            self._selections_history.append((request_id, best))

        for b in bundles:
            key = (request_id, b["bundle_id"])
            self._bundle_map[key] = b
            self._util_of[key] = b["utility"]
            edge_d = {}
            for e, d in b["edge_demands"].items():
                edge_d[tuple(sorted(e))] = d
            self._edge_of[key] = edge_d
            self._mem_of[key] = b["memory_demands"]
            self._latency_of[key] = b.get("latency", 0.0)
            self._fidelity_of[key] = b.get("fidelity", 0.0)

    def remove_request(self, request_id: str):
        if request_id in self.selections:
            self.selections.pop(request_id, None)
            self._active_requests.discard(request_id)
            if request_id in self._bundles:
                del self._bundles[request_id]

    def active_count(self) -> int:
        return len(self._active_requests)

    def _pick_best_feasible(self, request_id: str, bundles: List[dict],
                            current_selections: dict) -> Optional[str]:
        scored = []
        for b in bundles:
            feasible = True
            trial = dict(current_selections)
            trial[request_id] = b["bundle_id"]
            edge_load = self._compute_edge_load(trial)
            mem_load = self._compute_mem_load(trial)
            for edge, load in edge_load.items():
                if load > self.edge_capacities.get(edge, 0):
                    feasible = False
                    break
            if feasible:
                for node, load in mem_load.items():
                    if load > self.memory_capacities.get(node, 0):
                        feasible = False
                        break
            if feasible:
                scored.append((b["utility"], b["bundle_id"]))
        if scored:
            scored.sort(reverse=True)
            return scored[0][1]
        return None

    def _compute_edge_load(self, selections: dict) -> Dict[tuple, int]:
        load = defaultdict(int)
        for rid, bid in selections.items():
            if bid is None:
                continue
            key = (rid, bid)
            for edge, d in self._edge_of.get(key, {}).items():
                load[edge] += d
        return dict(load)

    def _compute_mem_load(self, selections: dict) -> Dict[str, int]:
        load = defaultdict(int)
        for rid, bid in selections.items():
            if bid is None:
                continue
            key = (rid, bid)
            for node, d in self._mem_of.get(key, {}).items():
                load[node] += d
        return dict(load)

    def _energy(self, selections: dict) -> float:
        total = 0.0
        edge_load = self._compute_edge_load(selections)
        mem_load = self._compute_mem_load(selections)

        for rid, bid in selections.items():
            if bid is None:
                continue
            key = (rid, bid)
            total -= self._util_of.get(key, 0.0)

        rid_bundle_count = defaultdict(int)
        for rid, bid in selections.items():
            if bid is not None:
                rid_bundle_count[rid] += 1
        for rid, cnt in rid_bundle_count.items():
            if cnt > 1:
                total += 100.0 * cnt * (cnt - 1) // 2

        for edge, load in edge_load.items():
            cap = self.edge_capacities.get(edge, 0)
            if load > cap:
                total += 10.0 * (load - cap) ** 2
            if self.congestion_weight > 0 and cap > 0:
                ratio = load / cap
                if ratio > self.congestion_threshold:
                    total += self.congestion_weight * (ratio - self.congestion_threshold) ** 2

        for node, load in mem_load.items():
            cap = self.memory_capacities.get(node, 0)
            if load > cap:
                total += 10.0 * (load - cap) ** 2

        if self.risk_weight > 0:
            for rid, bid in selections.items():
                if bid is None:
                    continue
                key = (rid, bid)
                wait = self._latency_of.get(key, 0.0) * self.hold_scale
                if self.use_fidelity_risk:
                    f = self._fidelity_of.get(key, 0.0)
                    decay = 1.0
                    if wait > 0:
                        t2 = self.risk_tau
                        t1v = 2.0 * t2
                        decay = 0.25 * (1.0 + 3.0 * math.exp(-wait / t2)
                                        * (1.0 + math.exp(-wait / t1v)) / 2.0)
                    total += self.risk_weight * (1.0 - f * decay)
                else:
                    total += self.risk_weight * (1.0 - math.exp(-wait / self.risk_tau))

        return total

    def local_sweep(self, n_steps: int = 100, temperature: float = 1.0):
        if not self._active_requests:
            return
        active_list = sorted(self._active_requests)
        for _ in range(n_steps):
            rid = self._rng.choice(active_list)
            bundles = self._bundles.get(rid, [])
            if not bundles:
                continue
            old_bid = self.selections.get(rid)
            pool = [b["bundle_id"] for b in bundles] + [None]
            pool = [p for p in pool if p != old_bid] + [old_bid]
            new_bid = self._rng.choice(pool)
            if new_bid == old_bid:
                continue
            candidate = dict(self.selections)
            candidate[rid] = new_bid
            e_old = self._energy(self.selections)
            e_new = self._energy(candidate)
            delta = e_new - e_old
            if delta < 0 or (temperature > 0 and self._rng.random() < math.exp(-delta / temperature)):
                self.selections[rid] = new_bid
                self._selections_history.append((rid, new_bid))

    def full_cooling_cycle(self, max_iterations: int = 2000,
                           initial_temperature: float = 10.0,
                           cooling_rate: float = 0.97,
                           steps_per_temperature: int = 50,
                           min_temperature: float = 1e-3):
        best = dict(self.selections)
        best_energy = self._energy(best)
        current = dict(self.selections)
        current_energy = best_energy
        temp = initial_temperature

        active_list = sorted(self._active_requests)
        if not active_list:
            return

        for _ in range(max_iterations):
            if temp < min_temperature:
                break
            rid = self._rng.choice(active_list)
            bundles = self._bundles.get(rid, [])
            if not bundles:
                continue
            old_bid = current.get(rid)
            if self._rng.random() < 0.3 or old_bid is None:
                pool = [b["bundle_id"] for b in bundles] + [None]
            else:
                pool = [b["bundle_id"] for b in bundles if b["bundle_id"] != old_bid] + [None]
            new_bid = self._rng.choice(pool)
            if new_bid == old_bid:
                continue
            candidate = dict(current)
            candidate[rid] = new_bid
            e_candidate = self._energy(candidate)
            delta = e_candidate - current_energy
            if delta < 0 or (temp > 0 and self._rng.random() < math.exp(-delta / temp)):
                current = candidate
                current_energy = e_candidate
                if current_energy < best_energy:
                    best = dict(current)
                    best_energy = current_energy
            if self._rng.random() < 1.0 / steps_per_temperature:
                temp *= cooling_rate

        self.selections = best

    def get_selected(self) -> List[Tuple[str, str]]:
        return [(rid, bid) for rid, bid in self.selections.items() if bid is not None]

    def get_energy(self) -> float:
        return self._energy(self.selections)

    def get_history(self) -> List[Tuple[str, Optional[str]]]:
        return list(self._selections_history)
