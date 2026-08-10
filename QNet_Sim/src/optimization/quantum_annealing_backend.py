"""Quantum-annealing backend for the QUBO pipeline (Wave 3, item 6).

A physical quantum annealer (D-Wave) exposes only a sparse hardware graph:
logical QUBO variables must be *minor-embedded* into chains of hardware qubits
connected by strong ferromagnetic couplings, the embedded problem is annealed,
and the chains are collapsed back to logical bits by majority vote.  This
module emulates that stack in full:

    1. ``build_hardware_graph`` -- a Chimera-like lattice (k_{t,t} unit cells);
    2. ``minor_embed`` -- greedy ``minorminer`` embedding of the logical QUBO;
    3. ``embed_bqm`` -- tile/chain construction of the embedded QUBO;
    4. ``default_chain_strength`` = max|h| + 2 max|J| (DWave's rule of thumb);
    5. ``PathIntegralAnnealingSampler`` -- simulated quantum annealing on the
       *embedded* hardware graph (the standard device emulator);
    6. ``unembed_by_majority`` -- deterministic chain-collapse.

``run_quantum_annealing_sweep`` then compares the QA backend against classical
simulated annealing (SA) and openjij's SQA emulator on utility, wall time and
chain-break fraction across problem sizes, and ``chain_strength_sweep``
justifies the strength choice.
"""

import math
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import dimod

try:
    import minorminer  # greedy minor-embedding solver
except ImportError:  # pragma: no cover
    minorminer = None


# ---------------------------------------------------------------------------
# Hardware graph
# ---------------------------------------------------------------------------
def build_hardware_graph(rows: int = 8, cols: int = 8, t: int = 4) -> "dict":
    """Chimera-like hardware graph: ``rows x cols`` unit cells, each a K_{t,t}.

    Returns an adjacency dict {node: set of neighbors} (networkx-free).
    The D-Wave Chimera topology is exactly this with ``t=4``.
    """
    adj: Dict[int, set] = {}
    for r in range(rows):
        for c in range(cols):
            base = (r * cols + c) * (2 * t)
            left = list(range(base, base + t))
            right = list(range(base + t, base + 2 * t))
            for u in left:
                adj.setdefault(u, set()).update(right)
                adj.setdefault(u, set()).add(u)
            for v in right:
                adj.setdefault(v, set()).update(left)
                adj.setdefault(v, set()).add(v)
            # vertical joins: right column of (r,c) to left column of (r+1,c)
            if r + 1 < rows:
                nb = ((r + 1) * cols + c) * (2 * t)
                for i in range(t):
                    adj[right[i]].add(nb + i)
                    adj.setdefault(nb + i, set()).add(right[i])
            # horizontal joins: right column of (r,c) to left column of (r,c+1)
            if c + 1 < cols:
                nb = (r * cols + (c + 1)) * (2 * t)
                for i in range(t):
                    adj[right[i]].add(nb + t + i)
                    adj.setdefault(nb + t + i, set()).add(right[i])
    return adj


def hardware_qubits(rows: int, cols: int, t: int = 4) -> int:
    return rows * cols * 2 * t


def _fit_hardware(n_qubits: int, t: int = 4) -> Tuple[int, int]:
    """Smallest square-ish lattice (t x t cells) with >= n_qubits qubits."""
    for rows in range(2, 17):
        for cols in range(2, 17):
            if rows * cols * 2 * t >= n_qubits:
                return rows, cols
    return 16, 16


# ---------------------------------------------------------------------------
# Minor embedding
# ---------------------------------------------------------------------------
def minor_embed(logical_adj: Dict, hardware_adj: Dict,
                tries: int = 10, timeout: int = 15,
                max_no_improvement: int = 300,
                random_seed: int = 42) -> Optional[Dict]:
    """Greedy minor-embedding of the logical graph into the hardware graph."""
    if minorminer is None:
        return None
    import networkx as nx
    S = nx.Graph()
    S.add_nodes_from(logical_adj)
    for u, nbrs in logical_adj.items():
        for v in nbrs:
            if u < v:
                S.add_edge(u, v)
    T = nx.Graph()
    T.add_nodes_from(hardware_adj)
    for u, nbrs in hardware_adj.items():
        for v in nbrs:
            if u < v:
                T.add_edge(u, v)
    return minorminer.find_embedding(S, T, tries=tries, timeout=timeout,
                                     max_no_improvement=max_no_improvement,
                                     random_seed=random_seed)


def default_chain_strength(bqm: dimod.BinaryQuadraticModel) -> float:
    """DWave's rule of thumb: max|h| + 2*max|J| (kept slightly above)."""
    hmax = max((abs(v) for v in bqm.linear.values()), default=0.0)
    jmax = max((abs(v) for v in bqm.quadratic.values()), default=0.0)
    return hmax + 2.0 * jmax + 0.1


# ---------------------------------------------------------------------------
# Tile embedding (exact for unbroken chains, QUBO/binary vartype)
# ---------------------------------------------------------------------------
def embed_bqm(bqm: dimod.BinaryQuadraticModel, embedding: Dict,
              chain_strength: float) -> dimod.BinaryQuadraticModel:
    """Embed ``bqm`` into hardware qubits with ferromagnetic chains.

    Construction (equal to DWave's tile method for binary vartype):
      * each logical linear bias ``h`` is split evenly over its chain;
      * every chain edge gets quad -2*cs and linear +cs on both ends, so a
        broken chain costs ``>= cs`` while a coherent chain costs 0;
      * every logical quadratic ``J`` is split evenly over all
        inter-chain edges (``|C_i| * |C_j|`` terms).
    """
    target = dimod.BQM(bqm.vartype)

    for v, h in bqm.linear.items():
        chain = embedding.get(v, [v])
        m = len(chain)
        for q in chain:
            target.add_linear(q, h / m)
        for a, b in zip(chain, chain[1:]):
            target.add_quadratic(a, b, -2.0 * chain_strength)
            target.add_linear(a, chain_strength)
            target.add_linear(b, chain_strength)

    for (u, v), J in bqm.quadratic.items():
        cu = embedding.get(u, [u])
        cv = embedding.get(v, [v])
        share = J / (len(cu) * len(cv))
        for a in cu:
            for b in cv:
                target.add_quadratic(a, b, share)

    return target


def unembed_by_majority(sample: Dict, embedding: Dict, n_logical: int,
                        rng: random.Random) -> Dict:
    """Collapse chain qubits to logical bits by (deterministic) majority vote.

    Ties are resolved to 0.  Any logical variable whose chain appears nowhere
    in the sample is treated as 0.
    """
    out = {}
    for logical, chain in embedding.items():
        votes = [sample.get(q, 0) for q in chain]
        ones = sum(votes)
        out[logical] = 1 if ones > len(chain) / 2 else 0
    return out


# ---------------------------------------------------------------------------
# Solver entry point
# ---------------------------------------------------------------------------
def quantum_anneal_solve(bundles: List[dict], edge_capacities: dict,
                         memory_capacities: dict, k: Optional[int] = 8,
                         num_reads: int = 50, num_sweeps: int = 400,
                         beta_range: Tuple[float, float] = (1e-4, 10.0),
                         chain_strength: Optional[float] = None,
                         rows: int = 0, cols: int = 0, t: int = 8,
                         seed: int = 42) -> Dict:
    """Solve the k-reduced QUBO with the emulated quantum-annealing backend."""
    from optimization.adaptive_qubo import AdaptiveCandidateSelector
    from optimization.qubo_optimizer import QUBOOptimizer

    selector = AdaptiveCandidateSelector(edge_capacities, memory_capacities,
                                         seed=seed)
    reduced = selector.select(bundles, k=k) if k else bundles
    optimizer = QUBOOptimizer(reduced, edge_capacities, memory_capacities)
    bqm = optimizer.to_bqm(penalty=100.0, edge_penalty=10.0,
                           memory_penalty=10.0,
                           congestion_penalty=0.0, memory_congestion_penalty=0.0)

    if chain_strength is None:
        chain_strength = default_chain_strength(bqm)

    n_hw = bqm.num_variables * 16
    if rows == 0 or cols == 0 or rows * cols * 2 * t < n_hw:
        rows, cols = _fit_hardware(n_hw, t=t)
    hardware = build_hardware_graph(rows, cols, t=t)

    logical_adj = {v: set() for v in bqm.variables}
    for u, v in bqm.quadratic:
        logical_adj.setdefault(u, set()).add(v)
        logical_adj.setdefault(v, set()).add(u)

    embedding = minor_embed(logical_adj, hardware, random_seed=seed)

    t0 = time.perf_counter()
    if not embedding:
        return {"k": k, "utility": 0.0, "served": 0,
                "n_qubo_variables": bqm.num_variables,
                "n_qubits": rows * cols * 2 * t,
                "chain_strength": chain_strength,
                "chain_break_fraction": 1.0,
                "embedded": False, "wall_time_s": time.perf_counter() - t0,
                "selected": []}

    embedded = embed_bqm(bqm, embedding, chain_strength)

    from dwave.samplers import PathIntegralAnnealingSampler
    response = PathIntegralAnnealingSampler().sample(
        embedded, num_reads=num_reads, num_sweeps=num_sweeps,
        beta_range=beta_range, seed=seed)

    rng = random.Random(seed)
    util = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}
    best_sel = []
    best_util = -1.0
    n_broken = 0
    for raw in response.samples():
        logical = unembed_by_majority(raw, embedding, bqm.num_variables, rng)
        selected = optimizer.decode_sample(logical, repair=True)
        u = sum(util.get(s, 0.0) for s in selected)
        if u > best_util:
            best_util = u
            best_sel = selected
        for chain in embedding.values():
            votes = [raw.get(q, 0) for q in chain]
            if sum(votes) != 0 and sum(votes) != len(chain):
                n_broken += 1
    total_chains = len(embedding) * len(response)
    elapsed = time.perf_counter() - t0

    return {
        "k": k,
        "n_bundles_in": len(bundles),
        "n_bundles_in_qubo": len(reduced),
        "n_qubo_variables": bqm.num_variables,
        "n_qubits": rows * cols * 2 * t,
        "n_chain_qubits": max(len(c) for c in embedding.values()),
        "chain_strength": chain_strength,
        "chain_break_fraction": n_broken / max(total_chains, 1),
        "embedded": True,
        "utility": best_util,
        "served": len(best_sel),
        "wall_time_s": elapsed,
        "selected": best_sel,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def chain_strength_sweep(bqm: dimod.BinaryQuadraticModel,
                         embedding: Dict, hardware_adj: Dict,
                         factors: Optional[List[float]] = None,
                         num_reads: int = 30, num_sweeps: int = 200,
                         beta_range: Tuple[float, float] = (1e-4, 10.0),
                         seed: int = 42) -> Dict:
    """Sweep chain strength around the default and report chain breaks.

    Returns the energy of the best sample, the chain-break fraction and the
    coherence of the returned sample per strength factor, so the report can
    justify ``max|h| + 2 max|J|`` empirically.
    """
    from dwave.samplers import PathIntegralAnnealingSampler
    if factors is None:
        factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    base = default_chain_strength(bqm)
    rows = []
    for f in factors:
        cs = base * f
        emb = embed_bqm(bqm, embedding, cs)
        resp = PathIntegralAnnealingSampler().sample(
            emb, num_reads=num_reads, num_sweeps=num_sweeps,
            beta_range=beta_range, seed=seed)
        energy = float(resp.record.energy.min())
        n_broken = 0
        for raw in resp.samples():
            for chain in embedding.values():
                votes = [raw.get(q, 0) for q in chain]
                if 0 < sum(votes) < len(chain):
                    n_broken += 1
        rows.append({
            "factor": f,
            "chain_strength": cs,
            "best_energy": energy,
            "chain_break_fraction": n_broken / (len(embedding) * num_reads),
        })
    return {"default_chain_strength": base, "rows": rows}


def run_quantum_annealing_sweep(topology_fn: Callable,
                                n_requests_list: Optional[List[int]] = None,
                                num_reads: int = 40, num_sweeps: int = 400,
                                k: int = 8, seed: int = 42) -> Dict:
    """QA backend vs classical SA / openjij SQA across problem sizes."""
    from experiments.instances import contention_sweep_instances
    from optimization.adaptive_qubo import (AdaptiveCandidateSelector,
                                            adaptive_qubo_solve)
    from optimization.openjij_solver import solve_sqa
    from optimization.qubo_optimizer import QUBOOptimizer

    if n_requests_list is None:
        n_requests_list = [6, 10, 14]

    insts = contention_sweep_instances(topology_fn, n_requests_list, seed=seed)
    rows = []
    for n_req in n_requests_list:
        inst = insts["n%d" % n_req]
        bundles, ec, mc = (inst["bundles"], inst["edge_capacities"],
                           inst["memory_capacities"])

        qa = quantum_anneal_solve(bundles, ec, mc, k=k, num_reads=num_reads,
                                  num_sweeps=num_sweeps, seed=seed)

        sa = adaptive_qubo_solve(bundles, ec, mc, k=k, num_reads=num_reads,
                                 seed=seed)

        selector = AdaptiveCandidateSelector(ec, mc, seed=seed)
        reduced = selector.select(bundles, k=k)
        opt = QUBOOptimizer(reduced, ec, mc)
        bqm = opt.to_bqm(penalty=100.0, edge_penalty=10.0,
                         memory_penalty=10.0)
        t0 = time.perf_counter()
        sqa_resp = solve_sqa(bqm, num_reads=num_reads, seed=seed)
        sqa_sel = opt.decode_sample(sqa_resp.first.sample, repair=True)
        sqa_t = time.perf_counter() - t0
        util = {(b["request_id"], b["bundle_id"]): b["utility"] for b in bundles}

        rows.append({
            "n_requests": n_req,
            "n_qubo_variables": bqm.num_variables,
            "qa_utility": qa["utility"],
            "qa_served": qa["served"],
            "qa_wall_time_s": qa["wall_time_s"],
            "qa_n_qubits": qa["n_qubits"],
            "qa_chain_break_fraction": qa["chain_break_fraction"],
            "qa_embedded": qa["embedded"],
            "sa_utility": sa["utility"],
            "sa_wall_time_s": sa["wall_time_s"],
            "sqa_utility": sum(util.get(s, 0.0) for s in sqa_sel),
            "sqa_wall_time_s": sqa_t,
        })
    return {"rows": rows}
