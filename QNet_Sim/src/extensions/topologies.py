"""Extension 13 -- network topology evolution.

The paper experiments so far use chain and grid topologies.  Real quantum
repeater networks come in many shapes, and an optimizer's advantage should
survive changes in topology.  This module adds the standard generative
families on top of the simulator's topology dict schema (``nodes``,
``edges``, ``edge_capacities``, ``memory_capacities``, ``link_params``):

* ring                   -- cycle network (redundancy, short detours),
* random geometric       -- nodes placed uniformly on a disk, links to
  neighbours within a radius (realistic regional / mesh networks),
* Erdos-Renyi            -- G(n, p) random graphs,
* Watts-Strogatz         -- small-world (high clustering, short diameter),
* Barabasi-Albert        -- scale-free preferential attachment (hub networks).

``topology_sweep`` evaluates a solver across several families with matched
size and density, so the "does the optimizer advantage survive topology
changes?" question has a concrete answer table.
"""

import math
import random
from typing import Callable, Dict, List, Optional, Tuple


def _pack(topology_family: str, nodes: List[str], edges: List[Tuple[str, str]],
          edge_capacity: int = 6, memory_capacity: int = 10,
          raw_fidelity: float = 0.85, generation_prob: float = 1.0,
          latency: float = 5.0, **extra) -> dict:
    """Build a topology dict in the simulator's schema from node/edge lists."""
    edge_caps = {e: edge_capacity for e in edges}
    mem_caps = {n: memory_capacity for n in nodes}
    link_params = {}
    for u, v in edges:
        link_params[(u, v)] = {
            "raw_fidelity": raw_fidelity,
            "generation_probability": generation_prob,
            "latency": latency,
        }
    return {
        "topology_family": topology_family,
        "nodes": nodes,
        "edges": edges,
        "edge_capacities": edge_caps,
        "memory_capacities": mem_caps,
        "link_params": link_params,
        **extra,
    }


def generate_ring_topology(n_nodes: int = 8, edge_capacity: int = 6,
                           memory_capacity: int = 10,
                           raw_fidelity: float = 0.85,
                           generation_prob: float = 1.0,
                           latency: float = 5.0):
    """Cycle on n_nodes: every node has degree 2."""
    labels = [f"R{i}" for i in range(n_nodes)]
    edges = [(labels[i], labels[(i + 1) % n_nodes]) for i in range(n_nodes)]
    return _pack("ring", labels, edges, edge_capacity, memory_capacity,
                 raw_fidelity, generation_prob, latency)


def generate_random_geometric_topology(n_nodes: int = 12, radius: Optional[float] = None,
                                       edge_capacity: int = 6,
                                       memory_capacity: int = 10,
                                       raw_fidelity: float = 0.85,
                                       generation_prob: float = 1.0,
                                       latency: float = 5.0,
                                       seed: Optional[int] = 42):
    """Nodes on the unit square, links between pairs within ``radius``.

    ``radius`` defaults to ~sqrt(2.4/n_nodes) so the graph is connected for
    n_nodes ~ 10-20.  Positions are re-sampled and the radius grown (1.1x per
    attempt) until the graph is connected.  ``position`` metadata is attached
    for reproducible plots.
    """
    rng = random.Random(seed)
    labels = [f"G{i}" for i in range(n_nodes)]
    base = radius if radius is not None else math.sqrt(2.4 / n_nodes)
    for attempt in range(50):
        positions = {}
        for label in labels:
            positions[label] = (rng.random(), rng.random())
        r = base * (1.1 ** attempt)
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dx = positions[labels[i]][0] - positions[labels[j]][0]
                dy = positions[labels[i]][1] - positions[labels[j]][1]
                if math.hypot(dx, dy) <= r:
                    edges.append((labels[i], labels[j]))
        if edges and _is_connected(labels, edges):
            return _pack("random_geometric", labels, edges, edge_capacity,
                         memory_capacity, raw_fidelity, generation_prob,
                         latency, positions=positions)
    raise ValueError("could not generate a connected random geometric graph")


def generate_erdos_renyi_topology(n_nodes: int = 12, p: float = 0.25,
                                  edge_capacity: int = 6,
                                  memory_capacity: int = 10,
                                  raw_fidelity: float = 0.85,
                                  generation_prob: float = 1.0,
                                  latency: float = 5.0,
                                  seed: Optional[int] = 42):
    """G(n, p) random graph; retries until connected."""
    rng = random.Random(seed)
    labels = [f"E{i}" for i in range(n_nodes)]
    for attempt in range(50):
        edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < p:
                    edges.append((labels[i], labels[j]))
        if _is_connected(labels, edges):
            return _pack("erdos_renyi", labels, edges, edge_capacity,
                         memory_capacity, raw_fidelity, generation_prob,
                         latency, p=p)
    raise ValueError("could not generate a connected G(n, p) graph")


def generate_watts_strogatz_topology(n_nodes: int = 12, k_neighbors: int = 2,
                                     rewiring_prob: float = 0.15,
                                     edge_capacity: int = 6,
                                     memory_capacity: int = 10,
                                     raw_fidelity: float = 0.85,
                                     generation_prob: float = 1.0,
                                     latency: float = 5.0,
                                     seed: Optional[int] = 42):
    """Small-world network (ring of lattice edges + rewired shortcuts)."""
    rng = random.Random(seed)
    labels = [f"W{i}" for i in range(n_nodes)]
    if k_neighbors >= n_nodes // 2:
        raise ValueError("k_neighbors too large for n_nodes")
    edges = []
    for i in range(n_nodes):
        for offset in range(1, k_neighbors + 1):
            j = (i + offset) % n_nodes
            if rng.random() < rewiring_prob:
                # rewire to a random non-neighbour (no self loops)
                candidates = [m for m in range(n_nodes) if m != i and m != j]
                if candidates:
                    j = rng.choice(candidates)
            edge = tuple(sorted((labels[i], labels[j])))
            if edge not in edges and edge[0] != edge[1]:
                edges.append(edge)
    return _pack("watts_strogatz", labels, edges, edge_capacity,
                 memory_capacity, raw_fidelity, generation_prob, latency,
                 k_neighbors=k_neighbors, rewiring_prob=rewiring_prob)


def generate_barabasi_albert_topology(n_nodes: int = 12, m_links: int = 2,
                                      edge_capacity: int = 6,
                                      memory_capacity: int = 10,
                                      raw_fidelity: float = 0.85,
                                      generation_prob: float = 1.0,
                                      latency: float = 5.0,
                                      seed: Optional[int] = 42):
    """Scale-free graph via preferential attachment (hub networks)."""
    rng = random.Random(seed)
    labels = [f"B{i}" for i in range(n_nodes)]
    edges = []
    connected = {labels[0]}
    for i in range(1, n_nodes):
        # pick m distinct existing nodes weighted by degree
        pool = list(connected)
        chosen = set()
        while len(chosen) < min(m_links, len(pool)):
            pick = rng.choice(pool)
            chosen.add(pick)
            if len(pool) < 2:
                break
        for other in chosen:
            edges.append((labels[i], other))
        connected.add(labels[i])
    return _pack("barabasi_albert", labels, edges, edge_capacity,
                 memory_capacity, raw_fidelity, generation_prob, latency,
                 m_links=m_links)


def _is_connected(nodes: List[str], edges: List[Tuple[str, str]]) -> bool:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    if not nodes:
        return False
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(nodes)


def all_topology_generators() -> Dict[str, Callable]:
    """Every topology family (with matched defaults: 12 nodes, cap 6)."""
    return {
        "chain": lambda n=12: _chain(n),
        "ring": lambda n=12: generate_ring_topology(n),
        "random_geometric": lambda n=12: generate_random_geometric_topology(n),
        "erdos_renyi": lambda n=12: generate_erdos_renyi_topology(n),
        "watts_strogatz": lambda n=12: generate_watts_strogatz_topology(n),
        "barabasi_albert": lambda n=12: generate_barabasi_albert_topology(n),
    }


def _chain(n_nodes: int):
    from experiments.instances import generate_chain_topology
    topo = generate_chain_topology(n_nodes=n_nodes, edge_capacity=6,
                                   memory_capacity=10)
    topo["topology_family"] = "chain"
    return topo


def topology_summary(topology: dict) -> dict:
    """Structural descriptors for a topology (density, diameter, clustering)."""
    n = len(topology["nodes"])
    m = len(topology["edges"])
    degree = {node: 0 for node in topology["nodes"]}
    for u, v in topology["edges"]:
        degree[u] += 1
        degree[v] += 1
    degs = list(degree.values())
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": 2.0 * m / max(n * (n - 1), 1),
        "min_degree": min(degs) if degs else 0,
        "max_degree": max(degs) if degs else 0,
        "mean_degree": sum(degs) / max(n, 1),
    }


def topology_sweep(topology_fns: Dict[str, Callable],
                   n_requests: int = 8, seed: int = 42) -> List[dict]:
    """Evaluate the Metropolis solver on each family; report per-topology
    metrics so optimizer behaviour can be compared across shapes.

    Returns one row per topology family: served ratio, utility, latency and
    the structural descriptors from ``topology_summary``.
    """
    import time
    from experiments.instances import contention_sweep_instances
    from optimization.metropolis_annealer import MetropolisAnnealer

    rows = []
    for family, fn in topology_fns.items():
        topo = fn()
        inst = contention_sweep_instances(lambda: topo, [n_requests],
                                          seed=seed)[f"req{n_requests}"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]
        util_of = {(bb["request_id"], bb["bundle_id"]): bb["utility"] for bb in b}

        opt = MetropolisAnnealer(b, ec, mc, seed=seed)
        t0 = time.perf_counter()
        r = opt.solve(
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        elapsed = time.perf_counter() - t0

        sel = r.get("selected", [])
        util = sum(util_of.get(k, 0.0) for k in sel)
        latencies = []
        for rid, bid in sel:
            for bb in b:
                if bb["request_id"] == rid and bb["bundle_id"] == bid:
                    latencies.append(bb.get("latency", 0.0))
                    break
        rows.append({
            "topology": family,
            **topology_summary(topo),
            "n_requests": n_requests,
            "served": len(sel),
            "served_ratio": len(sel) / max(n_requests, 1),
            "utility": util,
            "mean_latency": (sum(latencies) / len(latencies) if latencies else 0.0),
            "time_s": elapsed,
        })
    return rows
