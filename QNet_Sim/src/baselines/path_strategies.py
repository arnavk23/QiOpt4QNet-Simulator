from __future__ import annotations

import math
from typing import List, Optional

import networkx as nx

from network.network import QuantumNetwork
from network.request import Request
from routing.path_generator import PathGenerator


def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-9))


class CandidatePathGenerator:
    """Wraps PathGenerator with additional weighted-shortest-path strategies
    and returns a deduplicated union of candidates for a request."""

    def __init__(self, network: QuantumNetwork):
        self.network = network
        self._k_shortest = PathGenerator(network)

    def shortest_path(self, request: Request) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.network.graph, request.source, request.destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def k_shortest_paths(self, request: Request, k: int = 3) -> List[List[str]]:
        return self._k_shortest.get_k_shortest_paths(request, k=k)

    def highest_fidelity_path(self, request: Request) -> Optional[List[str]]:
        """Maximizes end-to-end raw fidelity via a -log(fidelity) shortest path
        (fidelities compound multiplicatively along a path, so this is the
        correct linearization for Dijkstra)."""
        return self._weighted_path(request, lambda e: -_safe_log(e.raw_fidelity))

    def highest_success_path(self, request: Request) -> Optional[List[str]]:
        return self._weighted_path(request, lambda e: -_safe_log(e.generation_probability))

    def lowest_latency_path(self, request: Request) -> Optional[List[str]]:
        return self._weighted_path(request, lambda e: e.latency)

    def capacity_aware_path(self, request: Request, min_capacity: int = 1) -> Optional[List[str]]:
        """Shortest path restricted to edges with at least `min_capacity`."""
        sub = nx.Graph()
        sub.add_nodes_from(self.network.graph.nodes())
        for u, v, data in self.network.graph.edges(data=True):
            if data["data"].capacity >= min_capacity:
                sub.add_edge(u, v)
        try:
            return nx.shortest_path(sub, request.source, request.destination)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _weighted_path(self, request: Request, weight_fn) -> Optional[List[str]]:
        try:
            return nx.shortest_path(
                self.network.graph, request.source, request.destination,
                weight=lambda u, v, d: weight_fn(d["data"]),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def generate_candidates(self, request: Request, k: int = 3, min_capacity: int = 1) -> List[List[str]]:
        """Union of every strategy above, deduplicated, order-preserving."""
        candidates: List[List[str]] = []
        seen = set()

        def add(path):
            if path and len(path) >= 2:
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)

        add(self.shortest_path(request))
        for p in self.k_shortest_paths(request, k=k):
            add(p)
        add(self.highest_fidelity_path(request))
        add(self.highest_success_path(request))
        add(self.lowest_latency_path(request))
        add(self.capacity_aware_path(request, min_capacity=min_capacity))

        return candidates
