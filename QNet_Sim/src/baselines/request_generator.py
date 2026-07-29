from __future__ import annotations

import random
from typing import List, Optional, Union

from network.network import QuantumNetwork
from network.request import Request

# Fraction of all node-pairs to turn into requests, by load level.
LOAD_PRESETS = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.6,
}


def generate_requests(
    network: QuantumNetwork,
    load: Union[str, int] = "medium",
    fidelity_range: tuple = (0.5, 0.9),
    weight_range: tuple = (0.5, 5.0),
    allow_repeat_pairs: bool = True,
    seed: Optional[int] = None,
) -> List[Request]:
    rng = random.Random(seed)
    nodes = list(network.graph.nodes())
    if len(nodes) < 2:
        return []

    if isinstance(load, str):
        frac = LOAD_PRESETS.get(load, LOAD_PRESETS["medium"])
        n_pairs = len(nodes) * (len(nodes) - 1) // 2
        n_requests = max(1, round(frac * n_pairs))
    else:
        n_requests = int(load)

    requests: List[Request] = []
    used_pairs = set()
    attempts, max_attempts = 0, n_requests * 50 + 100

    while len(requests) < n_requests and attempts < max_attempts:
        attempts += 1
        src, dst = rng.sample(nodes, 2)
        pair = tuple(sorted((src, dst)))
        if not allow_repeat_pairs and pair in used_pairs:
            continue
        used_pairs.add(pair)

        min_fidelity = round(rng.uniform(*fidelity_range), 3)
        weight = round(rng.uniform(*weight_range), 2)
        requests.append(Request(source=src, destination=dst, minimum_fidelity=min_fidelity, weight=weight))

    return requests
