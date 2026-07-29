from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from baselines.classical_baselines import BaseAllocator

FEATURE_KEYS = [
    "path_length", "fidelity", "success_probability", "latency",
    "bell_pair_cost", "memory_demand", "purification_rounds",
    "min_required_fidelity", "request_weight",
]


def load_dataset(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_training_matrix(records: List[dict]):
    X, y = [], []
    for rec in records:
        for c in rec["candidates"]:
            X.append([c[k] for k in FEATURE_KEYS])
            y.append(1.0 if c["selected"] else 0.0)
    return np.array(X), np.array(y)


class BundleRankingModel:
    """Scores candidate bundles; higher score = more likely to be part of
    the optimal (CP-SAT) selection. Trained on dataset_generator.py output."""

    def __init__(self, hidden_layer_sizes=(32, 16), random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000, random_state=random_state)

    def fit(self, records: List[dict]) -> "BundleRankingModel":
        X, y = build_training_matrix(records)
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, candidates: List[dict]) -> np.ndarray:
        X = np.array([[c[k] for k in FEATURE_KEYS] for c in candidates])
        return self.model.predict(self.scaler.transform(X))

    def score_bundles(self, bundles: List[dict]) -> Dict[Tuple[str, str], float]:
        """Convenience: score a list of Bundle.to_dict()-style dicts (must
        contain FEATURE_KEYS) keyed by (request_id, bundle_id)."""
        preds = self.score(bundles)
        return {(b["request_id"], b["bundle_id"]): float(s) for b, s in zip(bundles, preds)}


class _MLScoreDecoder(BaseAllocator):
    """Greedy decoder driven by external ML scores instead of true utility.
    Mirrors CongestionAwareGreedyAllocator's feasibility bookkeeping but a
    static (non-adaptive) ranking, since the ML score already encodes the
    model's belief about how "good" a bundle is."""

    name = "ml_ranking_decoder"

    def __init__(self, bundles, edge_capacities, memory_capacities,
                 scores: Dict[Tuple[str, str], float], seed=None):
        super().__init__(bundles, edge_capacities, memory_capacities, seed=seed)
        self.scores = scores

    def solve(self) -> dict:
        edge_load = defaultdict(int)
        mem_load = defaultdict(int)
        selections = {rid: None for rid in self.requests}
        remaining = set(self.requests)

        ordered = sorted(self.bundles, key=lambda b: self.scores[(b["request_id"], b["bundle_id"])], reverse=True)
        for b in ordered:
            rid = b["request_id"]
            if rid not in remaining:
                continue
            if self._fits(b, edge_load, mem_load):
                selections[rid] = b["bundle_id"]
                self._commit(b, edge_load, mem_load)
                remaining.discard(rid)

        return self._finalize(selections)


def feasibility_aware_decode(bundles: List[dict], scores: Dict[Tuple[str, str], float],
                              edge_capacities, memory_capacities) -> dict:
    """Selects the highest-scoring still-feasible candidate repeatedly,
    respecting one-bundle-per-request and capacity limits. `bundles` must
    be in the minimal optimizer format (bundle_id, request_id, path,
    edge_demands, memory_demands, utility)."""
    return _MLScoreDecoder(bundles, edge_capacities, memory_capacities, scores).solve()
