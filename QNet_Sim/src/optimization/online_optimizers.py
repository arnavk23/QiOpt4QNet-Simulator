"""Static vs online vs receding-horizon optimization (Extension 3).

Dynamic traffic (``routing.temporal_request``) lets us study how the same
network behaves under three control regimes on identical arrival traces:

* **static**       -- the whole trace is known up front and solved jointly;
* **online**       -- each request is committed irrevocably on arrival;
* **receding-horizon** -- requests are committed per window, but the solver
  may still rearrange the current window's requests as new ones arrive.

All three are built on the same :class:`JointRoutingScheduler`, so the only
difference is *information* and *commitment*, not solver internals.  Deadline
compliance (Extension 18) is reported for each regime via the weighted
on-time metric ``sum_r p_r * n_r * 1[T_r <= d_r]``.
"""

from typing import Callable, Dict, List, Optional

from routing.temporal_request import TemporalRequest, batch_trace_by_slot
from optimization.joint_scheduler import (
    JointRoutingScheduler, generate_temporal_bundles,
)


def _weighted_on_time(schedule: Dict) -> float:
    return sum(d["n_pairs"] * 1.0 * int(d["on_time"]) for d in schedule["decisions"])


def _metrics(schedule: Dict, label: str, wall_time_s: float) -> Dict:
    return {
        "regime": label,
        "served": schedule["served"],
        "served_ratio": schedule["served_ratio"],
        "on_time_ratio": schedule["on_time_ratio"],
        "mean_delivered_fidelity": schedule["mean_delivered_fidelity"],
        "mean_memory_utilization": schedule["mean_memory_utilization"],
        "weighted_on_time": _weighted_on_time(schedule),
        "wall_time_s": wall_time_s,
    }


def static_solve(edge_capacities: dict, memory_capacities: dict,
                 bundles_by_request: Dict[str, List[dict]],
                 tau_mem: float = 5.0, risk_weight: float = 2.0,
                 seed: int = 42, **anneal_kwargs) -> Dict:
    """Full-knowledge joint solve of the whole trace."""
    import time
    sched = JointRoutingScheduler(edge_capacities, memory_capacities,
                                  tau_mem=tau_mem, risk_weight=risk_weight,
                                  seed=seed)
    for rid, bs in bundles_by_request.items():
        sched.add_request(rid, bs)
    t0 = time.perf_counter()
    schedule = sched.solve(**anneal_kwargs)
    return _metrics(schedule, "static", time.perf_counter() - t0)


def online_solve(edge_capacities: dict, memory_capacities: dict,
                 trace: List[TemporalRequest],
                 bundles_by_request: Dict[str, List[dict]],
                 tau_mem: float = 5.0, risk_weight: float = 2.0,
                 seed: int = 42, **anneal_kwargs) -> Dict:
    """Each request committed irrevocably on arrival (greedy + one local move)."""
    import time
    sched = JointRoutingScheduler(edge_capacities, memory_capacities,
                                  tau_mem=tau_mem, risk_weight=risk_weight,
                                  seed=seed)
    committed: List[str] = []
    t0 = time.perf_counter()
    for req in trace:
        sched.add_request(req.request_id, bundles_by_request[req.request_id])
        sched.solve(movable=[req.request_id], max_iterations=50, n_restarts=1,
                    initial_temperature=1.0)
        committed.append(req.request_id)
    schedule = sched.schedule()
    return _metrics(schedule, "online", time.perf_counter() - t0)


def receding_horizon_solve(edge_capacities: dict, memory_capacities: dict,
                           trace: List[TemporalRequest],
                           bundles_by_request: Dict[str, List[dict]],
                           tau_mem: float = 5.0, risk_weight: float = 2.0,
                           window_size: int = 3, slot_duration: float = 1.0,
                           seed: int = 42, **anneal_kwargs) -> Dict:
    """Per-window re-optimization; current-window requests remain movable.

    Requests stay movable for ``window_size`` slots after arrival; once they
    leave the window they are committed.  This is the classic receding-horizon
    (model-predictive) scheme: it sees only the near future but may re-plan
    within it.
    """
    import time
    sched = JointRoutingScheduler(edge_capacities, memory_capacities,
                                  tau_mem=tau_mem, risk_weight=risk_weight,
                                  seed=seed)
    batches = batch_trace_by_slot(trace, slot_duration)
    pending: List[str] = []
    committed: List[str] = []
    t0 = time.perf_counter()
    for slot, batch in enumerate(batches):
        for req in batch:
            sched.add_request(req.request_id, bundles_by_request[req.request_id])
            pending.append(req.request_id)
        sched.solve(movable=list(pending), max_iterations=150, n_restarts=1)
        # commit requests that arrived at or before `slot - window_size + 1`
        horizon_start = slot - window_size + 1
        still_pending = []
        for rid in pending:
            req = next(r for r in trace if r.request_id == rid)
            if req.arrival / slot_duration <= horizon_start:
                committed.append(rid)
            else:
                still_pending.append(rid)
        pending = still_pending
    schedule = sched.schedule()
    return _metrics(schedule, "receding_horizon", time.perf_counter() - t0)


def run_regime_comparison(topology_fn: Callable, n_slots: int = 15,
                          mean_rate: float = 1.5, tau_mem: float = 5.0,
                          deadline_horizon: float = 6.0,
                          window_size: int = 3, seed: int = 42) -> Dict:
    """Compare the three control regimes on one dynamic arrival trace."""
    from routing.temporal_request import generate_dynamic_trace
    topo = topology_fn()
    trace = generate_dynamic_trace(topo, n_slots, mean_rate,
                                   deadline_horizon=deadline_horizon, seed=seed)
    bundles = generate_temporal_bundles(topo, trace)
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]

    rows = [
        static_solve(ec, mc, bundles, tau_mem=tau_mem, seed=seed),
        online_solve(ec, mc, trace, bundles, tau_mem=tau_mem, seed=seed),
        receding_horizon_solve(ec, mc, trace, bundles, tau_mem=tau_mem,
                               window_size=window_size, seed=seed),
    ]
    return {"n_requests": len(trace), "n_slots": n_slots, "rows": rows}
