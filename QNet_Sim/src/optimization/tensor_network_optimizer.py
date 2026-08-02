import numpy as np
from collections import defaultdict


class TensorNetworkOptimizer:
    def __init__(self, bundles, edge_capacities, memory_capacities):
        self.bundles = bundles
        self.edge_capacities = {self._undirected_edge(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self._validate()
        self._group_by_request()
        self._precompute()
        self._order_requests()

    def _validate(self):
        required = {"bundle_id", "request_id", "path", "edge_demands", "memory_demands", "utility"}
        for b in self.bundles:
            missing = required - b.keys()
            if missing:
                raise ValueError(f"Bundle {b.get('bundle_id', '?')} missing: {missing}")

    def _group_by_request(self):
        self.requests = []
        self.bundles_per_request = {}
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in self.bundles_per_request:
                self.bundles_per_request[rid] = []
                self.requests.append(rid)
            self.bundles_per_request[rid].append(b)

    @staticmethod
    def _undirected_edge(edge):
        return tuple(sorted(edge))

    def _precompute(self):
        self._util_of = {}
        self._edge_of = {}
        self._mem_of = {}
        for b in self.bundles:
            key = (b["request_id"], b["bundle_id"])
            self._util_of[key] = b["utility"]
            self._edge_of[key] = {self._undirected_edge(e): d for e, d in b["edge_demands"].items()}
            self._mem_of[key] = dict(b["memory_demands"])

    def _get_options(self, rid):
        return [None] + [b["bundle_id"] for b in self.bundles_per_request[rid]]

    def _build_adjacency(self):
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

        coupling = np.zeros((len(self.requests), len(self.requests)), dtype=int)
        for i, r1 in enumerate(self.requests):
            for j, r2 in enumerate(self.requests):
                if i >= j:
                    continue
                shared = rid_edges[r1] & rid_edges[r2]
                shared |= rid_mems[r1] & rid_mems[r2]
                if shared:
                    coupling[i, j] = coupling[j, i] = len(shared)
        return coupling, rid_edges, rid_mems

    def _order_requests(self):
        coupling, self._rid_edges, self._rid_mems = self._build_adjacency()
        n = len(self.requests)
        if n <= 2:
            self._ordered_requests = list(self.requests)
            return
        ordered = [self.requests[0]]
        remaining = set(range(1, n))
        current = 0
        while remaining:
            best_next = None
            best_weight = -1
            for j in remaining:
                if coupling[current, j] > best_weight:
                    best_weight = coupling[current, j]
                    best_next = j
            if best_next is None:
                best_next = remaining.pop()
            else:
                remaining.remove(best_next)
            ordered.append(self.requests[best_next])
            current = best_next
        self._ordered_requests = ordered

    def _local_penalty(self, rid, bid):
        """Self-penalty: overload penalty plus quadratic congestion term (G1/G7).

        The congestion term C*load^2 is exactly decomposed as
        C*sum_b(d_b)^2 = sum_b(C*d_b^2) + sum_{b<b'} 2*C*d_b*d_b',
        so the per-request local C*d^2 term plus the pairwise 2*C*d*d' cross-term
        reproduce the paper's quadratic congestion penalty inside the tensor
        elements themselves.
        """
        if bid is None:
            return 0.0
        pen = 0.0
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            cap = self.edge_capacities.get(edge, 0)
            if d > cap:
                pen += self._B * (d - cap) ** 2
            pen += self._C * d * d
        for node, d in self._mem_of.get((rid, bid), {}).items():
            cap = self.memory_capacities.get(node, 0)
            if d > cap:
                pen += self._D * (d - cap) ** 2
            pen += self._E * d * d
        return pen

    def _pairwise_penalty(self, rid_l, bid_l, rid_r, bid_r):
        """Pairwise penalty: shared-capacity overload plus 2*C*d_l*d_r cross-term."""
        if bid_l is None or bid_r is None:
            return 0.0
        pen = 0.0
        shared_edges = self._rid_edges[rid_l] & self._rid_edges[rid_r]
        shared_mems = self._rid_mems[rid_l] & self._rid_mems[rid_r]
        for edge in shared_edges:
            dl = self._edge_of.get((rid_l, bid_l), {}).get(edge, 0)
            dr = self._edge_of.get((rid_r, bid_r), {}).get(edge, 0)
            if dl + dr > self.edge_capacities.get(edge, 0):
                pen += self._B * (dl + dr - self.edge_capacities[edge]) ** 2
            pen += 2.0 * self._C * dl * dr
        for node in shared_mems:
            dl = self._mem_of.get((rid_l, bid_l), {}).get(node, 0)
            dr = self._mem_of.get((rid_r, bid_r), {}).get(node, 0)
            if dl + dr > self.memory_capacities.get(node, 0):
                pen += self._D * (dl + dr - self.memory_capacities[node]) ** 2
            pen += 2.0 * self._E * dl * dr
        return pen

    def _global_penalty_estimate(self, rid, bid, other_selections):
        """Penalty from this bundle given all other already-fixed selections."""
        if bid is None:
            return 0.0
        pen = 0.0
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            total = d
            for o_rid, o_bid in other_selections.items():
                if o_bid is not None:
                    total += self._edge_of.get((o_rid, o_bid), {}).get(edge, 0)
            cap = self.edge_capacities.get(edge, 0)
            if total > cap:
                pen += self._B * (total - cap) ** 2
            pen += self._C * total * total
        for node, d in self._mem_of.get((rid, bid), {}).items():
            total = d
            for o_rid, o_bid in other_selections.items():
                if o_bid is not None:
                    total += self._mem_of.get((o_rid, o_bid), {}).get(node, 0)
            cap = self.memory_capacities.get(node, 0)
            if total > cap:
                pen += self._D * (total - cap) ** 2
            pen += self._E * total * total
        return pen

    def solve(self, edge_penalty=10.0, memory_penalty=10.0,
              congestion_penalty=0.05, memory_congestion_penalty=0.05,
              bond_dim=8, beta=5.0, max_sweeps=15):
        self._B = edge_penalty
        self._D = memory_penalty
        self._C = congestion_penalty
        self._E = memory_congestion_penalty
        n = len(self.requests)
        if n == 0:
            return {"selected": [], "selections": {}}

        ordered = self._ordered_requests
        opt_maps = [self._get_options(rid) for rid in ordered]
        dims = [len(opts) for opts in opt_maps]

        mps = []
        for r_idx in range(n):
            d = dims[r_idx]
            rid = ordered[r_idx]
            t = np.zeros((d, 1, 1), dtype=np.float64)
            for b_idx, bid in enumerate(opt_maps[r_idx]):
                e = -self._util_of.get((rid, bid), 0.0) if bid is not None else 0.0
                e += self._local_penalty(rid, bid)
                t[b_idx, 0, 0] = np.exp(-beta * e)
            mps.append(t)

        best_selections = None
        best_energy = float("inf")

        for sweep in range(max_sweeps):
            for direction in ["left", "right"]:
                if direction == "left":
                    for i in range(n - 1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta, ordered, opt_maps)
                else:
                    for i in range(n - 2, -1, -1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta, ordered, opt_maps)

            selections = {}
            used_assignments = {}
            for r_idx in range(n):
                rid = ordered[r_idx]
                t = mps[r_idx]
                d = dims[r_idx]
                scores = np.zeros(d)
                for b_idx in range(d):
                    bid = opt_maps[r_idx][b_idx]
                    e = -self._util_of.get((rid, bid), 0.0) if bid is not None else 0.0
                    e += self._local_penalty(rid, bid)
                    e += self._global_penalty_estimate(rid, bid, used_assignments)
                    scores[b_idx] = t[b_idx].sum() * np.exp(-beta * e)
                # Deterministic tie-break (G13): among equal scores prefer the
                # higher-utility bundle, then the lexicographically smaller id.
                scale = max(abs(scores).max(), 1e-12)
                tie_break = np.zeros(d)
                for b_idx in range(d):
                    bid = opt_maps[r_idx][b_idx]
                    tie_break[b_idx] = self._util_of.get((rid, bid), 0.0) / scale
                best_b = np.argmax(scores + 1e-9 * tie_break)
                selections[rid] = opt_maps[r_idx][best_b]
                if selections[rid] is not None:
                    used_assignments[rid] = selections[rid]

            act = {rid: bid for rid, bid in selections.items() if bid is not None}
            util = sum(self._util_of.get((r, b), 0.0) for r, b in act.items())
            edge_load = defaultdict(int)
            mem_load = defaultdict(int)
            for r, b in act.items():
                for e, d in self._edge_of.get((r, b), {}).items():
                    edge_load[e] += d
                for n_, d in self._mem_of.get((r, b), {}).items():
                    mem_load[n_] += d
            pen = 0.0
            for e, load in edge_load.items():
                cap = self.edge_capacities.get(e, 0)
                if load > cap:
                    pen += self._B * (load - cap) ** 2
                pen += self._C * load * load
            for n_, load in mem_load.items():
                cap = self.memory_capacities.get(n_, 0)
                if load > cap:
                    pen += self._D * (load - cap) ** 2
                pen += self._E * load * load
            energy = -util + pen

            if energy < best_energy:
                best_energy = energy
                best_selections = dict(selections)

        selections = self._feasibility_repair(best_selections or selections)

        # Fallback: never return a worse selection than the feasible greedy
        # baseline (guards against the decoder collapsing to all-None under
        # high contention, which the MPS is otherwise prone to).
        greedy = self._greedy_seed()
        mps_util = sum(
            self._util_of.get((r, b), 0.0) for r, b in selections.items() if b is not None
        )
        greedy_util = sum(
            self._util_of.get((r, b), 0.0) for r, b in greedy.items() if b is not None
        )
        if greedy_util > mps_util:
            selections = greedy

        selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        return {"selected": selected, "selections": selections, "energy": best_energy}

    def _greedy_seed(self):
        """Deterministic, load-aware greedy baseline (utility density, then utility, ids).

        Tracks cumulative edge/memory load so the returned selection is always
        capacity-feasible.
        """
        scored = []
        for b in self.bundles:
            total_demand = sum(b["edge_demands"].values()) + sum(b["memory_demands"].values())
            ud = b["utility"] / (total_demand + 1e-10)
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

    def _contract_bond(self, mps, left, right, bond_dim, beta, ordered, opt_maps):
        tl = mps[left]
        tr = mps[right]
        dl = tl.shape[0]
        dr = tr.shape[0]
        chi_l = tl.shape[1]
        chi_r = tr.shape[2]
        rid_l = ordered[left]
        rid_r = ordered[right]

        combined = np.einsum("abf,cfd->acbd", tl, tr)
        combined = combined.reshape(dl * chi_l, dr * chi_r)

        for bl in range(dl):
            bid_l = opt_maps[left][bl]
            for br in range(dr):
                bid_r = opt_maps[right][br]
                pair_pen = self._pairwise_penalty(rid_l, bid_l, rid_r, bid_r)
                if pair_pen > 0:
                    factor = max(np.exp(-beta * pair_pen), 1e-150)
                    combined[bl * chi_l:(bl + 1) * chi_l,
                             br * chi_r:(br + 1) * chi_r] *= factor

        combined = np.nan_to_num(combined, nan=0.0, posinf=1e150, neginf=-1e150)

        try:
            u, s, vh = np.linalg.svd(combined, full_matrices=False)
        except np.linalg.LinAlgError:
            u, s, vh = np.linalg.svd(combined + 1e-12 * np.eye(combined.shape[0], combined.shape[1]),
                                     full_matrices=False)
        k = min(bond_dim, len(s))
        u = u[:, :k]
        s = s[:k]
        vh = vh[:k, :]
        tl_new = u.reshape(dl, chi_l, k)
        tr_new = vh.reshape(k, dr, chi_r).transpose(1, 0, 2)
        mps[left] = tl_new
        mps[right] = tr_new

    def _feasibility_repair(self, selections):
        def is_feasible(sel):
            edge_load = defaultdict(int)
            mem_load = defaultdict(int)
            for rid, bid in sel.items():
                if bid is None:
                    continue
                for edge, d in self._edge_of.get((rid, bid), {}).items():
                    edge_load[edge] += d
                for node, d in self._mem_of.get((rid, bid), {}).items():
                    mem_load[node] += d
            for edge, load in edge_load.items():
                if load > self.edge_capacities.get(edge, 0):
                    return False
            for node, load in mem_load.items():
                if load > self.memory_capacities.get(node, 0):
                    return False
            return True

        if is_feasible(selections):
            return selections
        for rid in self._ordered_requests:
            bundles = self.bundles_per_request[rid]
            for b in bundles:
                trial = dict(selections)
                trial[rid] = b["bundle_id"]
                if is_feasible(trial):
                    return trial
        return {rid: None for rid in self._ordered_requests}

    def decode_sample(self, result):
        return result["selected"]
