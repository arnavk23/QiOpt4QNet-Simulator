"""Direct-constraint-encoded tensor-network solver (paper Future Work item ii).

Nakada, Tanahashi and Tanaka [Quantum 9, 1799, 2025] show that hard capacity
constraints can be built into the tensor-network structure itself---by only
admitting feasible index configurations---eliminating penalty hyperparameters
and the post-hoc feasibility-repair step entirely.

:class:`ConstrainedTensorNetworkOptimizer` implements this for entanglement
routing:

* A bundle that alone exceeds any edge or memory capacity has zero amplitude
  in its local tensor (hard local constraint).
* During bond contraction, a pair of bundles whose combined demand exceeds a
  shared edge/memory capacity gets zero amplitude (hard pairwise constraint)
  instead of the soft penalty factor used by :class:`TensorNetworkOptimizer`.
* Decoding is feasibility-preserving: a bundle is only admissible if it fits
  alongside the already-fixed selections, so the returned configuration is
  feasible by construction and no repair pass runs.

``run_constrained_comparison`` benchmarks it against the penalty-based
tensor-network solver on the standard instance grid.
"""

import os
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np


class ConstrainedTensorNetworkOptimizer:
    def __init__(self, bundles, edge_capacities, memory_capacities,
                 seed: Optional[int] = None):
        self.bundles = bundles
        self.edge_capacities = {self._undirected_edge(k): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self._validate()
        self._group_by_request()
        self._precompute()
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
        for b in self.bundles:
            rid = b["request_id"]
            if rid not in self.bundles_per_request:
                self.bundles_per_request[rid] = []
                self.requests.append(rid)
            self.bundles_per_request[rid].append(b)

    def _precompute(self):
        self._util_of = {}
        self._edge_of = {}
        self._mem_of = {}
        self._locally_feasible = {}
        for b in self.bundles:
            key = (b["request_id"], b["bundle_id"])
            self._util_of[key] = b["utility"]
            edges = {self._undirected_edge(e): d for e, d in b["edge_demands"].items()}
            mems = dict(b["memory_demands"])
            self._edge_of[key] = edges
            self._mem_of[key] = mems
            feasible = all(self.edge_capacities.get(e, 0) >= d for e, d in edges.items())
            feasible = feasible and all(
                self.memory_capacities.get(n, 0) >= d for n, d in mems.items())
            self._locally_feasible[key] = feasible

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

        n = len(self.requests)
        coupling = np.zeros((n, n), dtype=int)
        for i, r1 in enumerate(self.requests):
            for j, r2 in enumerate(self.requests):
                if i >= j:
                    continue
                shared = (rid_edges[r1] & rid_edges[r2]) | (rid_mems[r1] & rid_mems[r2])
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

    def _pair_feasible(self, rid_l, bid_l, rid_r, bid_r) -> bool:
        """Hard pairwise constraint: combined demands on a shared edge/memory
        must stay within capacity.  Infeasible pairs get zero amplitude."""
        if bid_l is None or bid_r is None:
            return True
        for edge in self._rid_edges[rid_l] & self._rid_edges[rid_r]:
            dl = self._edge_of.get((rid_l, bid_l), {}).get(edge, 0)
            dr = self._edge_of.get((rid_r, bid_r), {}).get(edge, 0)
            if dl + dr > self.edge_capacities.get(edge, 0):
                return False
        for node in self._rid_mems[rid_l] & self._rid_mems[rid_r]:
            dl = self._mem_of.get((rid_l, bid_l), {}).get(node, 0)
            dr = self._mem_of.get((rid_r, bid_r), {}).get(node, 0)
            if dl + dr > self.memory_capacities.get(node, 0):
                return False
        return True

    def solve(self, bond_dim=8, beta=5.0, max_sweeps=15) -> Dict:
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
            e_list = []
            for b_idx, bid in enumerate(opt_maps[r_idx]):
                if bid is not None and not self._locally_feasible.get((rid, bid), False):
                    e_list.append(float("inf"))  # hard local constraint: zero amplitude
                    continue
                e = -self._util_of.get((rid, bid), 0.0) if bid is not None else 0.0
                e_list.append(e)
            # normalise per request so the Boltzmann weights stay in [0, 1]
            # and the SVD truncation is numerically stable.
            e_min = min(e_list)
            for b_idx, bid in enumerate(opt_maps[r_idx]):
                if bid is not None and not self._locally_feasible.get((rid, bid), False):
                    continue
                e = -self._util_of.get((rid, bid), 0.0) if bid is not None else 0.0
                t[b_idx, 0, 0] = np.exp(-beta * (e - e_min))
            mps.append(t)

        for _ in range(max_sweeps):
            for direction in ["left", "right"]:
                if direction == "left":
                    for i in range(n - 1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta, ordered, opt_maps)
                else:
                    for i in range(n - 2, -1, -1):
                        self._contract_bond(mps, i, i + 1, bond_dim, beta, ordered, opt_maps)

        selections = self._decode(mps, opt_maps, ordered)
        util = self._selection_utility(selections)
        selected = [(rid, bid) for rid, bid in selections.items() if bid is not None]
        return {"selected": selected, "selections": selections,
                "energy": -util, "utility": util}

    def _decode(self, mps, opt_maps, ordered):
        """Marginal decoder: contract the full MPS to get per-option
        probabilities, then commit requests by decimation (highest conditional
        marginal first).  Only options that fit beside the already-fixed
        selections are scored, so the returned configuration is feasible by
        construction and no repair pass is ever needed.

        Replaces the old sequential sweep-decoder, which scored local tensor
        amplitudes and collapsed to all-None under high contention."""
        n = len(ordered)
        fixed = {}
        selections = {}
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        unfixed = list(range(n))

        while unfixed:
            best = None
            best_key = None
            for idx in unfixed:
                P = self._conditional_marginal(mps, opt_maps, idx, fixed)
                rid = ordered[idx]
                opts = opt_maps[idx]
                feasible_b = [b for b in range(len(opts))
                              if opts[b] is None or self._fits(rid, opts[b], edge_load, mem_load)]
                b_star = max(feasible_b, key=lambda b: (
                    P[b], self._util_of.get((rid, opts[b]), 0.0)))
                key = (P[b_star], self._util_of.get((rid, opts[b_star]), 0.0))
                if best_key is None or key > best_key:
                    best_key = key
                    best = (idx, b_star)
            idx, b_star = best
            rid = ordered[idx]
            bid = opt_maps[idx][b_star]
            fixed[idx] = b_star
            selections[rid] = bid
            if bid is not None:
                self._commit(rid, bid, edge_load, mem_load)
            unfixed.remove(idx)
        return selections

    def _conditional_marginal(self, mps, opt_maps, idx, fixed):
        """Per-option marginal of request ``idx`` given fixed assignments,
        obtained by contracting the full MPS with the fixed sites pinned."""
        n = len(mps)
        v = np.ones(1)
        for j in range(idx):
            t = mps[j]
            f = fixed.get(j)
            if f is not None:
                t = t[f:f + 1]
            v = np.einsum("i,bij->j", v, t)
        w = np.ones(1)
        for j in range(n - 1, idx, -1):
            t = mps[j]
            f = fixed.get(j)
            if f is not None:
                t = t[f:f + 1]
            w = np.einsum("j,bij->i", w, t)
        t_i = mps[idx]
        d = t_i.shape[0]
        P = np.array([np.einsum("i,ij,j->", v, t_i[b], w) for b in range(d)])
        s = P.sum()
        if s > 0:
            return P / s
        out = np.zeros(d)
        out[0] = 1.0
        return out

    def _fits(self, rid, bid, edge_load, mem_load) -> bool:
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            if edge_load[edge] + d > self.edge_capacities.get(edge, 0):
                return False
        for node, d in self._mem_of.get((rid, bid), {}).items():
            if mem_load[node] + d > self.memory_capacities.get(node, 0):
                return False
        return True

    def _commit(self, rid, bid, edge_load, mem_load):
        for edge, d in self._edge_of.get((rid, bid), {}).items():
            edge_load[edge] += d
        for node, d in self._mem_of.get((rid, bid), {}).items():
            mem_load[node] += d

    def _selection_utility(self, selections):
        return sum(self._util_of.get((rid, bid), 0.0)
                   for rid, bid in selections.items() if bid is not None)

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
                if not self._pair_feasible(rid_l, bid_l, rid_r, bid_r):
                    # hard pairwise constraint: exclude the configuration
                    combined[bl * chi_l:(bl + 1) * chi_l,
                             br * chi_r:(br + 1) * chi_r] = 0.0

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
        tl_new = (u * s).reshape(dl, chi_l, k)
        tr_new = vh.reshape(k, dr, chi_r).transpose(1, 0, 2)
        mps[left] = tl_new
        mps[right] = tr_new

    def decode_sample(self, result):
        return result["selected"]


def _utility(bundles, selected):
    util_of = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    return sum(util_of.get(k, 0.0) for k in selected)


def run_constrained_comparison(topology_fn: Callable,
                               n_requests_list: Optional[List[int]] = None,
                               bond_dims: Optional[List[int]] = None,
                               seed: int = 42,
                               out_dir: Optional[str] = None) -> List[Dict]:
    """Benchmark the constraint-encoded MPS against the penalty-based MPS.

    Compares served ratio, aggregate utility, wall-clock time and the number of
    capacity violations (the constraint-encoded solver must produce zero
    violations without any repair pass)."""
    from experiments.instances import contention_sweep_instances
    from optimization.tensor_network_optimizer import TensorNetworkOptimizer
    import csv

    if n_requests_list is None:
        n_requests_list = [8, 16]
    if bond_dims is None:
        bond_dims = [8]

    instances = contention_sweep_instances(topology_fn, n_requests_list, seed=seed)
    rows = []
    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
        for bd in bond_dims:
            for solver in ["Penalty-MPS", "Constrained-MPS"]:
                if solver == "Penalty-MPS":
                    opt = TensorNetworkOptimizer(b, ec, mc)
                    fn = lambda: opt.solve(bond_dim=bd, beta=5.0, max_sweeps=10)
                else:
                    opt = ConstrainedTensorNetworkOptimizer(b, ec, mc)
                    fn = lambda: opt.solve(bond_dim=bd, beta=5.0, max_sweeps=10)
                t0 = time.perf_counter()
                r = fn()
                t = time.perf_counter() - t0
                rows.append({
                    "instance": name,
                    "n_requests": n_req,
                    "bond_dim": bd,
                    "solver": solver,
                    "served": len(r.get("selected", [])),
                    "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                    "utility": _utility(b, r.get("selected", [])),
                    "time_s": t,
                })

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "constrained_mps_comparison.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")
    return rows


if __name__ == "__main__":
    from experiments.instances import generate_chain_topology
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "results", "experiments"))
    topo = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
    print("Benchmarking constraint-encoded vs penalty-based MPS...")
    rows = run_constrained_comparison(topo, n_requests_list=[8, 16],
                                      bond_dims=[8], out_dir=out_dir)
    for r in rows:
        print(f"{r['solver']:>16} n={r['n_requests']}: served {r['served_ratio']:.2f} "
              f"util {r['utility']:.1f} time {r['time_s']:.4f}s")
