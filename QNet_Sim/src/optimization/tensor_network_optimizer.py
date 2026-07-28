import numpy as np
from collections import defaultdict


class TensorNetworkOptimizer:
    def __init__(self, bundles, edge_capacities, memory_capacities):
        self.bundles = bundles
        self.edge_capacities = {self._undirected_edge(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self._validate()
        self._group_by_request()
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

    def _undirected_edge(self, edge):
        return tuple(sorted(edge))

    def _build_adjacency(self):
        rid_edges = {}
        rid_mems = {}
        rid_to_idx = {rid: i for i, rid in enumerate(self.requests)}
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
                shared_edges = rid_edges[r1] & rid_edges[r2]
                shared_mems = rid_mems[r1] & rid_mems[r2]
                if shared_edges or shared_mems:
                    coupling[i, j] = coupling[j, i] = len(shared_edges) + len(shared_mems)
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

    def _build_tensors(self):
        self._tensors = []
        self._option_map = []
        for rid in self._ordered_requests:
            bundles = self.bundles_per_request[rid]
            options = [None] + [b["bundle_id"] for b in bundles]
            d = len(options)
            self._option_map.append(options)
            local_util = np.zeros(d)
            for i, bid in enumerate(options):
                if bid is not None:
                    local_util[i] = self._get_utility(rid, bid)

            self._tensors.append({"util": local_util, "options": options, "rid": rid})

    def _get_utility(self, rid, bid):
        for b in self.bundles:
            if b["request_id"] == rid and b["bundle_id"] == bid:
                return b["utility"]
        return 0.0

    def _get_edge_demands(self, rid, bid):
        for b in self.bundles:
            if b["request_id"] == rid and b["bundle_id"] == bid:
                return {self._undirected_edge(e): d for e, d in b["edge_demands"].items()}
        return {}

    def _get_mem_demands(self, rid, bid):
        for b in self.bundles:
            if b["request_id"] == rid and b["bundle_id"] == bid:
                return dict(b["memory_demands"])
        return {}

    def _contract_sweep(self, bond_dim, beta):
        n = len(self._ordered_requests)
        if n == 0:
            return {}

        mps = []
        for r_idx in range(n):
            d = len(self._option_map[r_idx])
            mps.append(np.ones((d, 1, 1), dtype=np.float64))

        for _sweep in range(10):
            for direction in ["left", "right"]:
                if direction == "left":
                    for i in range(n - 1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta)
                else:
                    for i in range(n - 2, -1, -1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta)

        selections = {}
        for r_idx in range(n):
            t = mps[r_idx]
            d = t.shape[0]
            local_energies = np.zeros(d)
            for b_idx in range(d):
                energy = -self._tensors[r_idx]["util"][b_idx]
                local_energies[b_idx] = energy

            if r_idx > 0:
                left_env = np.sum(t[:, :, :], axis=(1, 2))
                for b_idx in range(d):
                    if np.isfinite(left_env[b_idx]):
                        pass

            tie_weights = -local_energies
            sorted_idx = np.argsort(-tie_weights)
            for b_idx in sorted_idx:
                if b_idx == 0:
                    selected_bid = None
                else:
                    selected_bid = self._option_map[r_idx][b_idx]
                selections[self._ordered_requests[r_idx]] = selected_bid
                break

        return selections

    def _contract_bond(self, mps, left, right, bond_dim, beta):
        tl = mps[left]
        tr = mps[right]
        dl = tl.shape[0]
        dr = tr.shape[0]
        chi_l = tl.shape[1]
        chi_r = tr.shape[2]

        combined = np.einsum("abf,cfd->acbd", tl, tr)
        combined = combined.reshape(dl * chi_l, dr * chi_r)

        penalty_matrix = self._compute_coupling_penalty(left, right)

        if penalty_matrix is not None:
            for bl in range(dl):
                for br in range(dr):
                    combined[bl * chi_l:(bl + 1) * chi_l, br * chi_r:(br + 1) * chi_r] *= np.exp(
                        -beta * penalty_matrix[bl, br]
                    )

        u, s, vh = np.linalg.svd(combined, full_matrices=False)
        k = min(bond_dim, len(s))
        u = u[:, :k]
        s = s[:k]
        vh = vh[:k, :]

        tl_new = u.reshape(dl, chi_l, k)
        tr_new = vh.reshape(k, dr, chi_r).transpose(1, 0, 2)
        mps[left] = tl_new
        mps[right] = tr_new

    def _compute_coupling_penalty(self, left_idx, right_idx):
        rid_l = self._ordered_requests[left_idx]
        rid_r = self._ordered_requests[right_idx]
        options_l = self._option_map[left_idx]
        options_r = self._option_map[right_idx]
        dl = len(options_l)
        dr = len(options_r)

        matrix = np.zeros((dl, dr))

        shared_edges = self._rid_edges[rid_l] & self._rid_edges[rid_r]
        shared_mems = self._rid_mems[rid_l] & self._rid_mems[rid_r]

        if not shared_edges and not shared_mems:
            return None

        for bl in range(dl):
            bid_l = options_l[bl]
            if bid_l is None:
                edge_l = {}
                mem_l = {}
            else:
                edge_l = self._get_edge_demands(rid_l, bid_l)
                mem_l = self._get_mem_demands(rid_l, bid_l)

            for br in range(dr):
                bid_r = options_r[br]
                if bid_r is None:
                    edge_r = {}
                    mem_r = {}
                else:
                    edge_r = self._get_edge_demands(rid_r, bid_r)
                    mem_r = self._get_mem_demands(rid_r, bid_r)

                penalty = 0.0
                for edge in shared_edges:
                    if edge in edge_l and edge in edge_r:
                        cap = self.edge_capacities.get(edge, 0)
                        if edge_l[edge] + edge_r[edge] > cap:
                            penalty += self._B * (edge_l[edge] + edge_r[edge] - cap) ** 2
                for node in shared_mems:
                    if node in mem_l and node in mem_r:
                        cap = self.memory_capacities.get(node, 0)
                        if mem_l[node] + mem_r[node] > cap:
                            penalty += self._D * (mem_l[node] + mem_r[node] - cap) ** 2

                matrix[bl, br] = penalty

        return matrix

    def _feasibility_repair(self, selections):
        def is_feasible(sel):
            edge_load = defaultdict(int)
            mem_load = defaultdict(int)
            for rid, bid in sel.items():
                if bid is None:
                    continue
                for edge, d in self._get_edge_demands(rid, bid).items():
                    edge_load[edge] += d
                for node, d in self._get_mem_demands(rid, bid).items():
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

    def solve(self, edge_penalty=10.0, memory_penalty=10.0,
              bond_dim=8, beta=5.0):
        self._B = edge_penalty
        self._D = memory_penalty
        self._build_tensors()

        if len(self._ordered_requests) == 0:
            return {"selected": [], "selections": {}}

        selections = self._contract_sweep(bond_dim, beta)
        selections = self._feasibility_repair(selections)

        selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        return {
            "selected": selected,
            "selections": selections,
        }

    def decode_sample(self, result):
        return result["selected"]
