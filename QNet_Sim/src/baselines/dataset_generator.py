"""
Runs the CP-SAT exact allocator (falling back to the utility-per-resource
greedy if OR-Tools isn't installed) over many randomly generated
network/workload instances, and writes one JSON record per instance with
every candidate bundle's features plus whether the reference solver
selected it. This is the training data for the candidate-bundle ranking
model in ranking_model.py.
"""

from __future__ import annotations

import json
import random
from typing import Optional

from network import topology
from routing.bundle_generation import BundleGenerator

from baselines.request_generator import generate_requests
from baselines.path_strategies import CandidatePathGenerator
from baselines.classical_baselines import CPSATAllocator, CPSAT_AVAILABLE, UtilityPerResourceGreedyAllocator

TOPOLOGIES = {
    "chain": lambda n: topology.generate_chain(n),
    "grid": lambda n: topology.generate_grid(max(2, int(round(n ** 0.5))), max(2, int(round(n ** 0.5)))),
    "random": lambda n: topology.generate_random(n, p=0.3),
}


def _edge_caps(network) -> dict:
    return {tuple(sorted((u, v))): d["data"].capacity for u, v, d in network.graph.edges(data=True)}


def _mem_caps(network) -> dict:
    return {node: d["data"].memory_capacity for node, d in network.graph.nodes(data=True)}


def generate_instance(n_nodes: int = 8, topology_type: str = "random", load: str = "medium",
                       k_paths: int = 3, seed: Optional[int] = None):
    net = TOPOLOGIES.get(topology_type, TOPOLOGIES["random"])(n_nodes)
    requests = generate_requests(net, load=load, seed=seed)

    path_gen = CandidatePathGenerator(net)
    bundle_gen = BundleGenerator(net)

    all_bundles = []
    for req in requests:
        candidate_paths = path_gen.generate_candidates(req, k=k_paths)
        all_bundles.extend(bundle_gen.generate_bundles(req, candidate_paths))

    return net, requests, all_bundles


def solve_reference(bundles, edge_caps, mem_caps) -> dict:
    """CP-SAT exact solution if OR-Tools is available, else the strongest
    classical greedy as a fallback oracle."""
    opt_dicts = [b.to_optimizer_dict() for b in bundles]
    if CPSAT_AVAILABLE:
        allocator = CPSATAllocator(opt_dicts, edge_caps, mem_caps)
    else:
        allocator = UtilityPerResourceGreedyAllocator(opt_dicts, edge_caps, mem_caps)
    return allocator.solve()


def bundle_features(bundle) -> dict:
    """Flat feature vector consumed by sthe ranking model."""
    return {
        "path_length": len(bundle.path) - 1,
        "fidelity": bundle.fidelity,
        "success_probability": bundle.success_probability,
        "latency": bundle.latency,
        "bell_pair_cost": bundle.bell_pair_cost,
        "memory_demand": bundle.memory_demand,
        "purification_rounds": bundle.purification_rounds,
        "utility": bundle.utility,
        "min_required_fidelity": bundle.request.minimum_fidelity,
        "request_weight": bundle.request.weight,
    }


def generate_dataset(n_instances: int = 50, out_path: str = "dataset.jsonl",
                      n_nodes: int = 8, topology_type: str = "random",
                      load: str = "medium", k_paths: int = 3) -> str:
    with open(out_path, "w") as f:
        for i in range(n_instances):
            net, requests, bundles = generate_instance(
                n_nodes=n_nodes, topology_type=topology_type, load=load, k_paths=k_paths, seed=i,
            )
            if not bundles:
                continue

            edge_caps, mem_caps = _edge_caps(net), _mem_caps(net)
            result = solve_reference(bundles, edge_caps, mem_caps)
            selected_keys = set(result["selected"])

            record = {
                "instance_id": i,
                "seed": i,
                "n_nodes": net.graph.number_of_nodes(),
                "n_edges": net.graph.number_of_edges(),
                "n_requests": len(requests),
                "objective": result.get("total_utility"),
                "candidates": [
                    {
                        **bundle_features(b),
                        "request_id": b.request_id,
                        "bundle_id": b.bundle_id,
                        "selected": (b.request_id, b.bundle_id) in selected_keys,
                    }
                    for b in bundles
                ],
            }
            f.write(json.dumps(record) + "\n")

    return out_path


if __name__ == "__main__":
    path = generate_dataset(n_instances=20, out_path="dataset.jsonl", n_nodes=8, load="medium")
    print(f"Wrote dataset to {path}")
