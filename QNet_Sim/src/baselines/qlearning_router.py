"""Q-learning adaptive router (paper Future Work item iii).

A reinforcement-learning baseline for the streaming/adaptive routing setting:
requests arrive over time and a router must decide each request's bundle (or
reject it) given the current network occupancy.  We implement a tabular
Q-learning agent whose state is the quantized global resource pressure
(edge-occupancy and memory-occupancy bins) and whose action is the bundle
choice for the arriving request.

The policy is trained on episodes built from a synthetic Poisson arrival trace
and evaluated against the streaming annealer on the same trace.  This answers
"when should one prefer RL over Hamiltonian-based solvers for online routing"
by measuring the utility gap and per-decision latency.
"""

import os
import random
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _edge_load_of(selections, edge_of):
    load = defaultdict(int)
    for rid, bid in selections.items():
        if bid is None:
            continue
        for e, d in edge_of.get((rid, bid), {}).items():
            load[e] += d
    return load


def _mem_load_of(selections, mem_of):
    load = defaultdict(int)
    for rid, bid in selections.items():
        if bid is None:
            continue
        for n, d in mem_of.get((rid, bid), {}).items():
            load[n] += d
    return load


class QLearningRouter:
    """Tabular Q-learning router over quantized resource-pressure states."""

    def __init__(self, edge_capacities, memory_capacities, n_bins: int = 3,
                 lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1,
                 seed: Optional[int] = None):
        self.edge_capacities = {tuple(sorted(k)): v for k, v in edge_capacities.items()}
        self.memory_capacities = memory_capacities
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.q: Dict[Tuple, float] = defaultdict(float)
        self.edge_of: Dict[Tuple[str, str], dict] = {}
        self.mem_of: Dict[Tuple[str, str], dict] = {}
        self.util_of: Dict[Tuple[str, str], float] = {}
        self.training_episodes = 0

    # -- feature plumbing ------------------------------------------------
    def _register_bundles(self, request_id: str, bundles: List[dict]):
        for b in bundles:
            key = (request_id, b["bundle_id"])
            self.util_of[key] = b["utility"]
            self.edge_of[key] = {tuple(sorted(e)): d for e, d in b["edge_demands"].items()}
            self.mem_of[key] = dict(b["memory_demands"])

    def _state(self, selections: dict) -> Tuple[int, int]:
        """Quantize global resource pressure into (edge_bin, memory_bin)."""
        edge_load = _edge_load_of(selections, self.edge_of)
        mem_load = _mem_load_of(selections, self.mem_of)
        ratios = [load / self.edge_capacities.get(e, 1)
                  for e, load in edge_load.items() if self.edge_capacities.get(e, 1) > 0]
        mem_ratios = [load / self.memory_capacities.get(n, 1)
                      for n, load in mem_load.items() if self.memory_capacities.get(n, 1) > 0]
        avg_edge = (sum(ratios) / len(ratios)) if ratios else 0.0
        max_mem = max(mem_ratios) if mem_ratios else 0.0
        b_e = min(self.n_bins - 1, int(avg_edge * self.n_bins))
        b_m = min(self.n_bins - 1, int(max_mem * self.n_bins))
        return (b_e, b_m)

    def _fits(self, rid, bid, selections) -> bool:
        trial = dict(selections)
        trial[rid] = bid
        edge_load = _edge_load_of(trial, self.edge_of)
        for e, load in edge_load.items():
            if load > self.edge_capacities.get(e, 0):
                return False
        mem_load = _mem_load_of(trial, self.mem_of)
        for n, load in mem_load.items():
            if load > self.memory_capacities.get(n, 0):
                return False
        return True

    def choose_action(self, rid: str, bundles: List[dict], selections: dict):
        """Epsilon-greedy choice over feasible bundles (or reject)."""
        self._register_bundles(rid, bundles)
        state = self._state(selections)
        feasible = [b["bundle_id"] for b in bundles
                    if self._fits(rid, b["bundle_id"], selections)]
        actions = feasible + [None]
        if not feasible:
            return None
        if self._rng.random() < self.epsilon:
            return self._rng.choice(actions)
        best = max(actions, key=lambda a: self.q.get((state, a), 0.0))
        return best

    def _update(self, state, action, reward, next_state):
        best_next = max((self.q[(next_state, a)] for a in self._all_actions(next_state)),
                        default=0.0)
        self.q[(state, action)] += self.lr * (
            reward + self.gamma * best_next - self.q[(state, action)])

    def _all_actions(self, state):
        # actions seen in this state so far (bundle ids or None)
        return {a for (s, a) in self.q if s == state}

    def train_episode(self, trace: List[Dict[str, List[dict]]],
                      full_capacity_penalty: float = 50.0) -> Tuple[float, float]:
        """One RL episode: process the arrival trace, return (served, utility)."""
        selections: Dict[str, Optional[str]] = {}
        total_reward = 0.0
        served = 0
        for slot in trace:
            for rid, bundles in slot.items():
                self._register_bundles(rid, bundles)
                state = self._state(selections)
                action = self.choose_action(rid, bundles, selections)
                reward = self.util_of.get((rid, action), 0.0) if action is not None else 0.0
                if action is not None and not self._fits(rid, action, selections):
                    reward = -full_capacity_penalty
                    action = None
                selections[rid] = action
                if action is not None:
                    served += 1
                next_state = self._state(selections)
                self._update(state, action, reward, next_state)
                total_reward += reward
        self.training_episodes += 1
        return served, total_reward

    def evaluate(self, trace: List[Dict[str, List[dict]]]) -> Tuple[int, float, int]:
        """Pure greedy replay of the trace (no exploration, no learning)."""
        selections: Dict[str, Optional[str]] = {}
        total_utility = 0.0
        served = 0
        for slot in trace:
            for rid, bundles in slot.items():
                self._register_bundles(rid, bundles)
                feasible = [b["bundle_id"] for b in bundles
                            if self._fits(rid, b["bundle_id"], selections)]
                best = None
                if feasible:
                    state = self._state(selections)
                    best = max(feasible, key=lambda a: self.q.get((state, a), 0.0))
                selections[rid] = best
                if best is not None:
                    served += 1
                    total_utility += self.util_of.get((rid, best), 0.0)
        return served, total_utility, self.training_episodes


def _utility_sum(bundles_by_rid, selected):
    util = 0.0
    for rid, bid in selected:
        if bid is None:
            continue
        for b in bundles_by_rid.get(rid, []):
            if b["bundle_id"] == bid:
                util += b["utility"]
                break
    return util


def run_rl_comparison(topology_fn: Callable, n_slots: int = 20,
                      mean_rate: float = 1.5, episodes: int = 30,
                      seed: int = 42, out_dir: Optional[str] = None) -> Dict:
    """Train a tabular Q-learning router and compare it against the streaming
    annealer on a held-out Poisson trace.

    Returns aggregate served ratio / utility / per-decision latency for both
    routers and writes per-episode learning progress to ``out_dir``."""
    from optimization.streaming_annealer import StreamingAnnealer
    from optimization.time_dependent_optimizer import _poisson_trace
    import csv, time

    topo = topology_fn()
    ec, mc = topo["edge_capacities"], topo["memory_capacities"]
    rng = random.Random(seed)
    all_traces = []
    for _ in range(episodes + 1):
        all_traces.append(_poisson_trace(topology_fn, n_slots, mean_rate, seed=rng.randint(0, 2 ** 31 - 1)))
    train_traces = all_traces[:episodes]
    test_trace = all_traces[-1]

    # Train the Q-learning router
    router = QLearningRouter(ec, mc, seed=seed)
    progress = []
    for ep in range(episodes):
        served, reward = router.train_episode(train_traces[ep])
        progress.append({"episode": ep, "served": served, "reward": reward})

    # Evaluate RL greedily
    t0 = time.perf_counter()
    rl_served, rl_utility, _ = router.evaluate(test_trace)
    rl_time = time.perf_counter() - t0

    # Streaming annealer baseline on the same test trace
    sa = StreamingAnnealer(ec, mc, seed=seed)
    bundles_by_rid = {}
    t0 = time.perf_counter()
    for slot in test_trace:
        for rid, bundles in slot.items():
            bundles_by_rid[rid] = bundles
            sa.add_request(rid, bundles)
            sa.local_sweep(n_steps=50, temperature=2.0)
    sa_time = time.perf_counter() - t0
    sa_sel = sa.get_selected()

    n_requests = sum(len(slot) for slot in test_trace)

    result = {
        "n_requests": n_requests,
        "qlearning": {
            "served": rl_served,
            "served_ratio": rl_served / max(n_requests, 1),
            "utility": rl_utility,
            "wall_time_s": rl_time,
        },
        "streaming_annealer": {
            "served": len(sa_sel),
            "served_ratio": len(sa_sel) / max(n_requests, 1),
            "utility": _utility_sum(bundles_by_rid, sa_sel),
            "wall_time_s": sa_time,
        },
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "qlearning_comparison.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(progress[0].keys()))
            w.writeheader()
            w.writerows(progress)
        print(f"Wrote {path}")
    result["progress"] = progress
    return result


if __name__ == "__main__":
    from experiments.instances import generate_chain_topology
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "results", "experiments"))
    topo = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
    print("Training Q-learning router and comparing against streaming annealer...")
    res = run_rl_comparison(topo, n_slots=10, mean_rate=1.0, episodes=10, out_dir=out_dir)
    for name in ["qlearning", "streaming_annealer"]:
        r = res[name]
        print(f"{name:>18}: served {r['served']}/{res['n_requests']} "
              f"({r['served_ratio']:.3f}), utility {r['utility']:.1f}, "
              f"time {r['wall_time_s']:.4f}s")
