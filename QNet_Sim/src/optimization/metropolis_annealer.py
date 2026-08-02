import math
import random
from collections import defaultdict


class MetropolisAnnealer:
    def __init__(self, bundles, edge_capacities, memory_capacities, seed=None):
        self.bundles = bundles
        self.edge_capacities = {self._undirected_edge(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self._rng = random.Random(seed)
        self._validate()
        self._group_by_request()
        self._A = 100.0
        self._B = 10.0
        self._D = 10.0
        self._C = 0.05
        self._E = 0.05

    @staticmethod
    def _undirected_edge(edge):
        return tuple(sorted(edge))

    def _validate(self):
        required = {"bundle_id", "request_id", "path", "edge_demands", "memory_demands", "utility"}
        for b in self.bundles:
            missing = required - b.keys()
            if missing:
                raise ValueError(f"Bundle {b.get('bundle_id', '?')} missing: {missing}")

    def _group_by_request(self):
        self.requests = []
        self.bundles_per_request = {}
        self._util_of = {}
        self._edge_of = {}
        self._mem_of = {}
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in self.bundles_per_request:
                self.bundles_per_request[rid] = []
                self.requests.append(rid)
            self.bundles_per_request[rid].append(b)
            key = (rid, b["bundle_id"])
            self._util_of[key] = b["utility"]
            edge_demands = {}
            for e, d in b["edge_demands"].items():
                edge_demands[tuple(sorted(e))] = d
            self._edge_of[key] = edge_demands
            self._mem_of[key] = b["memory_demands"]

    def _edge_pen(self, e, load):
        """Edge penalty: overload term (paper) + quadratic congestion term (paper G1)."""
        cap = self.edge_capacities.get(e, 0)
        pen = 0.0
        if load > cap:
            pen += self._B * (load - cap) ** 2
        pen += self._C * load * load
        return pen

    def _mem_pen(self, n, load):
        """Memory penalty: overload term + quadratic congestion term (our extension)."""
        cap = self.memory_capacities.get(n, 0)
        pen = 0.0
        if load > cap:
            pen += self._D * (load - cap) ** 2
        pen += self._E * load * load
        return pen

    def _energy(self, selections):
        total_utility = 0.0
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)

        for rid, bid in selections.items():
            if bid is None:
                continue
            key = (rid, bid)
            total_utility += self._util_of[key]
            for edge, d in self._edge_of[key].items():
                edge_load[edge] += d
            for node, d in self._mem_of[key].items():
                mem_load[node] += d

        energy = -total_utility
        for edge, load in edge_load.items():
            energy += self._edge_pen(edge, load)
        for node, load in mem_load.items():
            energy += self._mem_pen(node, load)
        return energy

    def _loads(self, selections):
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        for rid, bid in selections.items():
            if bid is None:
                continue
            key = (rid, bid)
            for edge, d in self._edge_of[key].items():
                edge_load[edge] += d
            for node, d in self._mem_of[key].items():
                mem_load[node] += d
        return edge_load, mem_load

    def _energy_delta(self, rid, old_bid, new_bid, edge_load, mem_load):
        """Incremental O(L) energy delta for changing request rid's bundle (G2).

        Only edges/nodes touched by either the old or new bundle are recomputed,
        so cost scales with path length L, not network size E.
        """
        if old_bid == new_bid:
            return 0.0

        delta = 0.0

        old_u = self._util_of.get((rid, old_bid), 0.0) if old_bid is not None else 0.0
        new_u = self._util_of.get((rid, new_bid), 0.0) if new_bid is not None else 0.0
        delta += old_u - new_u

        old_edges = self._edge_of.get((rid, old_bid), {}) if old_bid is not None else {}
        new_edges = self._edge_of.get((rid, new_bid), {}) if new_bid is not None else {}
        for e in set(old_edges) | set(new_edges):
            d_old = old_edges.get(e, 0)
            d_new = new_edges.get(e, 0)
            load_old = edge_load.get(e, 0)
            load_new = load_old - d_old + d_new
            delta += self._edge_pen(e, load_new) - self._edge_pen(e, load_old)

        old_mems = self._mem_of.get((rid, old_bid), {}) if old_bid is not None else {}
        new_mems = self._mem_of.get((rid, new_bid), {}) if new_bid is not None else {}
        for n in set(old_mems) | set(new_mems):
            d_old = old_mems.get(n, 0)
            d_new = new_mems.get(n, 0)
            load_old = mem_load.get(n, 0)
            load_new = load_old - d_old + d_new
            delta += self._mem_pen(n, load_new) - self._mem_pen(n, load_old)

        return delta

    def _apply_delta(self, rid, old_bid, new_bid, edge_load, mem_load):
        old_edges = self._edge_of.get((rid, old_bid), {}) if old_bid is not None else {}
        new_edges = self._edge_of.get((rid, new_bid), {}) if new_bid is not None else {}
        for e in set(old_edges) | set(new_edges):
            d_old = old_edges.get(e, 0)
            d_new = new_edges.get(e, 0)
            edge_load[e] += d_new - d_old
        old_mems = self._mem_of.get((rid, old_bid), {}) if old_bid is not None else {}
        new_mems = self._mem_of.get((rid, new_bid), {}) if new_bid is not None else {}
        for n in set(old_mems) | set(new_mems):
            d_old = old_mems.get(n, 0)
            d_new = new_mems.get(n, 0)
            mem_load[n] += d_new - d_old

    def _greedy_seed(self):
        scored = []
        for b in self.bundles:
            total_demand = sum(b["edge_demands"].values()) + sum(b["memory_demands"].values())
            ud = b["utility"] / (total_demand + 1e-10)
            # Deterministic tie-break (G13): higher utility, then request id, then bundle id
            scored.append((ud, -b["utility"], b["request_id"], b["bundle_id"], b))
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
        selections = {}
        assigned = set()
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        for _, _, _, _, b in scored:
            rid = b["request_id"]
            if rid in assigned:
                continue
            feasible = True
            for edge, d in b["edge_demands"].items():
                e = tuple(sorted(edge))
                if edge_load[e] + d > self.edge_capacities.get(e, 0):
                    feasible = False
                    break
            if feasible:
                for node, d in b["memory_demands"].items():
                    if mem_load[node] + d > self.memory_capacities.get(node, 0):
                        feasible = False
                        break
            if feasible:
                selections[rid] = b["bundle_id"]
                assigned.add(rid)
                for edge, d in b["edge_demands"].items():
                    edge_load[tuple(sorted(edge))] += d
                for node, d in b["memory_demands"].items():
                    mem_load[node] += d
        for rid in self.requests:
            if rid not in selections:
                selections[rid] = None
        return selections

    def _random_seed(self):
        selections = {}
        for rid in self.requests:
            bundles = self.bundles_per_request[rid]
            pool = [b["bundle_id"] for b in bundles] + [None]
            selections[rid] = self._rng.choice(pool)
        return selections

    def _metropolis_chain(self, initial, beta, steps, track_best=True):
        """Run a Metropolis chain at fixed inverse temperature beta.

        Uses incremental O(L) energy updates: the full energy is computed once,
        then each proposal is evaluated via local load deltas (G2).
        """
        current = dict(initial)
        edge_load, mem_load = self._loads(current)
        current_energy = self._energy(current)
        best = dict(current) if track_best else None
        best_energy = current_energy if track_best else None
        accepts = 0

        for _ in range(steps):
            rid = self._rng.choice(self.requests)
            bundles = self.bundles_per_request[rid]
            old_bid = current[rid]
            if self._rng.random() < 0.3 or old_bid is None:
                pool = [b["bundle_id"] for b in bundles] + [None]
            else:
                pool = [b["bundle_id"] for b in bundles if b["bundle_id"] != old_bid] + [None]

            new_bid = self._rng.choice(pool)
            if new_bid == old_bid:
                continue

            delta = self._energy_delta(rid, old_bid, new_bid, edge_load, mem_load)

            if delta < 0 or (beta > 0 and self._rng.random() < math.exp(-beta * delta)):
                self._apply_delta(rid, old_bid, new_bid, edge_load, mem_load)
                current[rid] = new_bid
                current_energy += delta
                accepts += 1
                if track_best and current_energy < best_energy:
                    best = dict(current)
                    best_energy = current_energy

        rate = accepts / max(steps, 1)
        return current, current_energy, best, best_energy, rate

    def solve(self, penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
              congestion_penalty=0.05, memory_congestion_penalty=0.05,
              max_iterations=5000, initial_temperature=10.0, cooling_rate=0.99,
              steps_per_temperature=50, min_temperature=1e-3, patience=20,
              n_restarts=5, target_accept_rate=0.25, initial_selections=None):
        self._A = penalty
        self._B = edge_penalty
        self._D = memory_penalty
        self._C = congestion_penalty
        self._E = memory_congestion_penalty

        best_overall = None
        best_energy_overall = float("inf")

        for restart in range(n_restarts):
            if restart == 0 and initial_selections is not None:
                current = dict(initial_selections)
            elif restart == 0:
                current = self._greedy_seed()
            else:
                current = self._random_seed()

            current_energy = self._energy(current)
            best = dict(current)
            best_energy = current_energy
            temperature = initial_temperature
            stalled = 0
            window_size = max(10, steps_per_temperature)

            for iteration in range(max_iterations):
                if temperature < min_temperature:
                    break

                _, _, _, _, accept_rate = self._metropolis_chain(
                    current, 1.0 / temperature, window_size, track_best=False
                )

                current_chain, current_energy, best_chain, best_chain_energy, _ = self._metropolis_chain(
                    current, 1.0 / temperature, steps_per_temperature, track_best=True
                )
                current = current_chain
                current_energy = current_energy

                if best_chain is not None and best_chain_energy < best_energy:
                    best = best_chain
                    best_energy = best_chain_energy
                    stalled = 0
                else:
                    stalled += 1

                if accept_rate < 0.05:
                    temperature = min(initial_temperature, temperature * 2.0)
                elif accept_rate < target_accept_rate * 0.5:
                    temperature *= 1.5
                elif accept_rate > target_accept_rate * 2.0:
                    temperature *= 0.85
                else:
                    temperature *= cooling_rate

                if stalled >= patience:
                    temperature = min(initial_temperature, temperature * 1.5)
                    stalled = 0

            if best_energy < best_energy_overall:
                best_overall = dict(best)
                best_energy_overall = best_energy

        selected = [(rid, bid) for rid, bid in best_overall.items() if bid is not None]
        return {
            "selected": selected,
            "energy": best_energy_overall,
            "selections": best_overall,
        }

    def decode_sample(self, result):
        return result["selected"]
