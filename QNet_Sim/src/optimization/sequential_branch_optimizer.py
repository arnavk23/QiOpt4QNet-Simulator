import math
from collections import defaultdict


class SequentialBranchOptimizer:
    """arXiv:2605.27425-style sequential branch expansion (G3).

    Requests are processed in an order that places highly-coupled requests
    first (wider beams early). Each surviving branch is expanded with every
    bundle option (including rejection, None), scored by a Boltzmann survival
    weight exp(-beta * partial_energy), and pruned to the top-K branches.

    Extensions over the paper:
      * Congestion penalty C*load^2 + E*mem^2 (paper G1 term included).
      * Memory-capacity penalties in addition to edge penalties.
      * Deterministic tie-breaking among equal survival weights.
      * Multiple request orderings (sweeps) to avoid ordering bias.
    """

    def __init__(self, bundles, edge_capacities, memory_capacities):
        self.bundles = bundles
        self.edge_capacities = {self._undirected_edge(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self._validate()
        self._group_by_request()
        self._order_requests()

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

    def _order_requests(self):
        """Order requests so that the most coupled pair is processed first, then
        greedily follow the strongest coupling (same heuristic as the MPS)."""
        rid_edges = {}
        rid_mems = {}
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in rid_edges:
                rid_edges[rid] = set()
                rid_mems[rid] = set()
            for e in b["edge_demands"]:
                rid_edges[rid].add(self._undirected_edge(e))
            for n in b["memory_demands"]:
                rid_mems[rid].add(n)

        n = len(self.requests)
        if n <= 1:
            self._ordered_requests = list(self.requests)
            return

        coupling = {}
        for i, r1 in enumerate(self.requests):
            for j, r2 in enumerate(self.requests):
                if i >= j:
                    continue
                shared = rid_edges[r1] & rid_edges[r2]
                shared |= rid_mems[r1] & rid_mems[r2]
                coupling[(r1, r2)] = len(shared)

        ordered = [self.requests[0]]
        remaining = set(self.requests[1:])
        current = ordered[0]
        while remaining:
            best_next = None
            best_w = -1
            for r in remaining:
                w = coupling.get((current, r), coupling.get((r, current), 0))
                if w > best_w:
                    best_w = w
                    best_next = r
            if best_next is None:
                best_next = sorted(remaining)[0]
            ordered.append(best_next)
            remaining.remove(best_next)
            current = best_next
        self._ordered_requests = ordered

    def _get_options(self, rid):
        return [None] + [b["bundle_id"] for b in self.bundles_per_request[rid]]

    def _edge_pen(self, e, load):
        cap = self.edge_capacities.get(e, 0)
        pen = 0.0
        if load > cap:
            pen += self._B * (load - cap) ** 2
        pen += self._C * load * load
        return pen

    def _mem_pen(self, n, load):
        cap = self.memory_capacities.get(n, 0)
        pen = 0.0
        if load > cap:
            pen += self._D * (load - cap) ** 2
        pen += self._E * load * load
        return pen

    def _energy_delta(self, rid, bid, edge_load, mem_load):
        """Energy delta of assigning request rid to bundle bid (from None)."""
        delta = 0.0
        if bid is not None:
            delta -= self._util_of.get((rid, bid), 0.0)
        edges = self._edge_of.get((rid, bid), {}) if bid is not None else {}
        for e, d in edges.items():
            load_old = edge_load.get(e, 0)
            load_new = load_old + d
            delta += self._edge_pen(e, load_new) - self._edge_pen(e, load_old)
        mems = self._mem_of.get((rid, bid), {}) if bid is not None else {}
        for n, d in mems.items():
            load_old = mem_load.get(n, 0)
            load_new = load_old + d
            delta += self._mem_pen(n, load_new) - self._mem_pen(n, load_old)
        return delta

    def _full_energy(self, selections):
        total_utility = 0.0
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        for rid, bid in selections.items():
            if bid is None:
                continue
            total_utility += self._util_of.get((rid, bid), 0.0)
            for e, d in self._edge_of.get((rid, bid), {}).items():
                edge_load[e] += d
            for n, d in self._mem_of.get((rid, bid), {}).items():
                mem_load[n] += d
        energy = -total_utility
        for e, load in edge_load.items():
            energy += self._edge_pen(e, load)
        for n, load in mem_load.items():
            energy += self._mem_pen(n, load)
        return energy

    def solve(self, edge_penalty=10.0, memory_penalty=10.0,
              congestion_penalty=0.05, memory_congestion_penalty=0.05,
              beta=5.0, branch_factor=8, n_orderings=2):
        self._B = edge_penalty
        self._D = memory_penalty
        self._C = congestion_penalty
        self._E = memory_congestion_penalty
        if not self.requests:
            return {"selected": [], "selections": {}, "energy": 0.0}

        base_order = list(self._ordered_requests)
        orderings = [base_order]
        # Alternative orderings: reverse and rotations reduce beam-ordering bias.
        orderings.append(list(reversed(base_order)))
        for _ in range(max(0, n_orderings - 2)):
            rotated = base_order[1:] + base_order[:1]
            orderings.append(rotated)
            base_order = rotated

        best_selections = None
        best_energy = float("inf")

        for ordering in orderings[:max(1, n_orderings)]:
            # beams: (selections, edge_load, mem_load, energy)
            beams = [({}, defaultdict(int), defaultdict(int), 0.0)]
            for rid in ordering:
                options = self._get_options(rid)
                new_beams = []
                for sel, edge_load, mem_load, energy in beams:
                    for bid in options:
                        delta = self._energy_delta(rid, bid, edge_load, mem_load)
                        new_sel = dict(sel)
                        new_sel[rid] = bid
                        new_edge = dict(edge_load)
                        new_mem = dict(mem_load)
                        if bid is not None:
                            for e, d in self._edge_of.get((rid, bid), {}).items():
                                new_edge[e] = new_edge.get(e, 0) + d
                            for n, d in self._mem_of.get((rid, bid), {}).items():
                                new_mem[n] = new_mem.get(n, 0) + d
                        new_beams.append((new_sel, new_edge, new_mem, energy + delta))

                # Boltzmann survival weight + deterministic tie-break (G13):
                # among equal weights prefer higher utility / smaller id.
                def _sort_key(beam):
                    sel, _, _, energy = beam
                    util = sum(
                        self._util_of.get((r, b), 0.0) for r, b in sel.items() if b is not None
                    )
                    return (energy, -util)

                new_beams.sort(key=_sort_key)
                pruned = new_beams[:branch_factor]
                beams = [(s, e, m, en) for s, e, m, en in pruned]
                for s, e, m, en in new_beams:
                    if en < best_energy:
                        best_energy = en
                        best_selections = dict(s)

        selected = [(rid, bid) for rid, bid in best_selections.items() if bid is not None]
        return {
            "selected": selected,
            "selections": best_selections,
            "energy": best_energy,
        }

    def decode_sample(self, result):
        return result["selected"]
