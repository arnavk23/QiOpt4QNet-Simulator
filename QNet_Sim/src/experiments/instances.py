"""
experiments/instances.py

Deterministic (fixed-parameter, non-randomized) topology generators and a
request-count "contention sweep" instance builder, used by
baselines/gnn_baseline.py to produce training/val/test data.

Unlike network.topology's generate_chain/generate_grid (which draw link
fidelity/probability/latency from random ranges by default), the
generators here take fixed values so that calling the same topology_fn()
twice yields an identical network -- required because gnn_baseline.py
calls the topology function once for the raw topology dict (for GNN
tensorization) and again inside contention_sweep_instances (to build
bundles), and the two need to agree.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from network.network import QuantumNetwork
from network.node import QuantumNode
from network.link import QuantumLink

from baselines.request_generator import generate_requests
from baselines.path_strategies import CandidatePathGenerator
from routing.bundle_generation import BundleGenerator


def _add_link(net: QuantumNetwork, u: str, v: str, edge_capacity: int, raw_fidelity: float,
              generation_prob: float, latency: float, distance: float,
              edges: list, link_params: dict) -> None:
    net.add_link(QuantumLink(u, v, distance, generation_prob, raw_fidelity, latency, edge_capacity))
    e = tuple(sorted((u, v)))
    edges.append(e)
    link_params[e] = {
        "raw_fidelity": raw_fidelity,
        "generation_probability": generation_prob,
        "latency": latency,
    }


def generate_chain_topology(n_nodes: int, edge_capacity: int, memory_capacity: int,
                             raw_fidelity: float, generation_prob: float, latency: float,
                             distance: float = 25.0) -> dict:
    net = QuantumNetwork()
    nodes = [f"Node_{i}" for i in range(n_nodes)]
    for nid in nodes:
        net.add_node(QuantumNode(nid, memory_capacity))

    edges: list = []
    link_params: dict = {}
    for i in range(n_nodes - 1):
        _add_link(net, nodes[i], nodes[i + 1], edge_capacity, raw_fidelity,
                   generation_prob, latency, distance, edges, link_params)

    return {
        "nodes": nodes,
        "edges": edges,
        "link_params": link_params,
        "edge_capacities": {e: edge_capacity for e in edges},
        "memory_capacities": {n: memory_capacity for n in nodes},
        "network": net,
    }


def generate_grid_topology(rows: int, cols: int, edge_capacity: int, memory_capacity: int,
                            raw_fidelity: float, generation_prob: float, latency: float,
                            distance: float = 25.0) -> dict:
    net = QuantumNetwork()
    nodes = [f"Node_{i}_{j}" for i in range(rows) for j in range(cols)]
    for nid in nodes:
        net.add_node(QuantumNode(nid, memory_capacity))

    edges: list = []
    link_params: dict = {}
    for i in range(rows):
        for j in range(cols - 1):
            _add_link(net, f"Node_{i}_{j}", f"Node_{i}_{j + 1}", edge_capacity, raw_fidelity,
                       generation_prob, latency, distance, edges, link_params)
    for i in range(rows - 1):
        for j in range(cols):
            _add_link(net, f"Node_{i}_{j}", f"Node_{i + 1}_{j}", edge_capacity, raw_fidelity,
                       generation_prob, latency, distance, edges, link_params)

    return {
        "nodes": nodes,
        "edges": edges,
        "link_params": link_params,
        "edge_capacities": {e: edge_capacity for e in edges},
        "memory_capacities": {n: memory_capacity for n in nodes},
        "network": net,
    }


def contention_sweep_instances(topology_fn: Callable[[], dict], request_counts: Sequence[int],
                                seed: int, k_paths: int = 3) -> Dict[str, dict]:
    """For each request count, builds a *fresh* network from topology_fn(),
    generates that many requests, generates candidate paths + bundles, and
    returns {workload_name: {"bundles": [...], "edge_capacities": {...},
    "memory_capacities": {...}}}.

    topology_fn must be deterministic (fixed link parameters, no randomness)
    so this network matches the one gnn_baseline.py separately builds for
    GNN node/edge feature tensorization.
    """
    out: Dict[str, dict] = {}
    for n_requests in request_counts:
        topo = topology_fn()
        net = topo["network"]
        requests = generate_requests(net, load=n_requests, seed=seed)

        path_gen = CandidatePathGenerator(net)
        bundle_gen = BundleGenerator(net)

        bundles = []
        for req in requests:
            candidate_paths = path_gen.generate_candidates(req, k=k_paths)
            bundles.extend(bundle_gen.generate_bundles(req, candidate_paths))

        out[f"req{n_requests}"] = {
            "bundles": [b.to_dict() for b in bundles],
            "edge_capacities": dict(topo["edge_capacities"]),
            "memory_capacities": dict(topo["memory_capacities"]),
        }
    return out
 