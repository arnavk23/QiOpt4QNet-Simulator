"""Dynamic traffic model (paper Extension 3 & 18).

Requests no longer arrive as a single static batch: they have an arrival
time, a soft deadline, a priority, and a required number of Bell pairs.
``generate_dynamic_trace`` produces a Poisson arrival process of
:class:`TemporalRequest` objects, which the static / online / receding-horizon
routers in ``optimization.online_optimizers`` consume.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Dict, List, Tuple


@dataclass
class TemporalRequest:
    """A request with a temporal profile.

    ``arrival``      -- earliest time the request can start,
    ``deadline``     -- latest acceptable completion time (T_r <= d_r),
    ``priority``     -- weight multiplier in the objective (w_r),
    ``n_pairs``      -- number of end-to-end Bell pairs requested,
    ``max_latency``  -- hard upper bound on the delivery latency.
    """

    request_id: str
    source: str
    destination: str
    minimum_fidelity: float
    weight: float = 1.0
    arrival: float = 0.0
    deadline: float = float("inf")
    priority: float = 1.0
    n_pairs: int = 1
    max_latency: float = float("inf")

    def to_request(self):
        """Adapt to the legacy ``network.request.Request`` interface."""
        from network.request import Request
        return Request(source=self.source, destination=self.destination,
                       minimum_fidelity=self.minimum_fidelity,
                       weight=self.weight, request_id=self.request_id)

    def completion_ok(self, completion_time: float) -> bool:
        return completion_time <= self.deadline

    def slack(self, completion_time: float) -> float:
        return self.deadline - completion_time

    def __repr__(self):
        return (f"TemporalRequest({self.request_id}: {self.source}->{self.destination}, "
                f"arr={self.arrival}, d={self.deadline:.1f}, p={self.priority:.2f}, "
                f"Fmin={self.minimum_fidelity}, pairs={self.n_pairs})")


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    count = 0
    remaining = 1.0
    while True:
        u = rng.random()
        if u <= 0.0:
            u = 1e-12
        remaining -= -math.log(u) / lam
        if remaining < 0.0:
            break
        count += 1
    return count


def generate_dynamic_trace(topology: dict, n_slots: int, mean_rate: float,
                           slot_duration: float = 1.0,
                           deadline_horizon: float = 6.0,
                           priority_bounds: Tuple[float, float] = (0.5, 1.0),
                           fidelity_bounds: Tuple[float, float] = (0.5, 0.85),
                           n_pairs_bounds: Tuple[int, int] = (1, 3),
                           weight_bounds: Tuple[float, float] = (10.0, 60.0),
                           seed: int = 42) -> List[TemporalRequest]:
    """Poisson arrival of temporal requests over ``n_slots`` time slots.

    Each request gets ``arrival = slot * slot_duration`` and a soft deadline
    ``arrival + U[0.5, 1] * deadline_horizon`` (so a fraction of requests are
    inherently hard to meet --- the interesting regime for deadline-aware
    scheduling).  Returns requests sorted by arrival.
    """
    rng = random.Random(seed)
    nodes = list(topology["nodes"])
    requests: List[TemporalRequest] = []
    rid = 0
    for slot in range(n_slots):
        arrival = slot * slot_duration
        for _ in range(_poisson(rng, mean_rate)):
            src, dst = rng.sample(nodes, 2)
            fmin = rng.uniform(*fidelity_bounds)
            priority = rng.uniform(*priority_bounds)
            n_pairs = rng.randint(*n_pairs_bounds)
            weight = rng.uniform(*weight_bounds)
            horizon = rng.uniform(0.5, 1.0) * deadline_horizon
            deadline = arrival + horizon
            requests.append(TemporalRequest(
                request_id=f"req_{rid}",
                source=src, destination=dst,
                minimum_fidelity=fmin,
                weight=weight,
                arrival=arrival,
                deadline=deadline,
                priority=priority,
                n_pairs=n_pairs,
                max_latency=horizon,
            ))
            rid += 1
    requests.sort(key=lambda r: r.arrival)
    return requests


def batch_trace_by_slot(trace: List[TemporalRequest],
                        slot_duration: float = 1.0) -> List[List[TemporalRequest]]:
    """Group a sorted request trace into per-slot batches."""
    if not trace:
        return []
    n_slots = int(math.floor(trace[-1].arrival / slot_duration)) + 1
    batches: List[List[TemporalRequest]] = [[] for _ in range(n_slots)]
    for r in trace:
        idx = min(int(math.floor(r.arrival / slot_duration)), n_slots - 1)
        batches[idx].append(r)
    return batches
