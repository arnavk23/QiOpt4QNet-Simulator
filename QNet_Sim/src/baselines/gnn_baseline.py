"""
QiOpt4QNet: edge-aware GNN baseline for route/purification bundle allocation.

Recommended repository location:
    QNet_Sim/src/baselines/gnn_baseline.py

Run from QNet_Sim:
    PYTHONPATH=src python src/baselines/gnn_baseline.py \
        --train-seeds 20 --val-seeds 5 --test-seeds 10 --device auto

Paper-style run:
    PYTHONPATH=src python src/baselines/gnn_baseline.py \
        --train-seeds 40 --val-seeds 10 --test-seeds 20 \
        --request-counts 8 16 24 --full-topologies \
        --epochs 300 --hidden-dim 64 --layers 3 --device auto

The baseline:
  1. generates complete QiOpt4QNet instances using experiments.instances;
  2. solves small/medium instances with CP-SAT to get oracle labels;
  3. trains an edge-aware GraphSAGE-style model in pure PyTorch (no PyG);
  4. scores candidate route/purification bundles;
  5. decodes scores with a hard-feasibility-aware greedy allocator;
  6. evaluates held-out instances against CP-SAT and utility-density greedy.

Important: data are split by complete instance, never by individual bundles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow direct execution from QNet_Sim/src/baselines/.
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.instances import (  # noqa: E402
    contention_sweep_instances,
    generate_chain_topology,
    generate_grid_topology,
)
from baselines.classical_baselines import CPSATAllocator, CPSAT_AVAILABLE  # noqa: E402

BundleKey = Tuple[str, str]

NODE_DIM = 6
EDGE_DIM = 4
BUNDLE_DIM = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def undirected(edge) -> Tuple[str, str]:
    return tuple(sorted(tuple(edge)))


def bundle_key(bundle: dict) -> BundleKey:
    return str(bundle["request_id"]), str(bundle["bundle_id"])


@dataclass
class GNNInstance:
    instance_id: str
    topology_name: str
    seed: int
    topology: dict
    bundles: List[dict]
    edge_capacities: dict
    memory_capacities: dict
    oracle_selected: List[BundleKey]
    oracle_utility: float
    oracle_optimal: bool
    oracle_status: str


@dataclass
class TensorInstance:
    node_x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    bundle_x: torch.Tensor
    paths: List[List[int]]
    path_edge_ids: List[List[int]]
    keys: List[BundleKey]
    labels: torch.Tensor
    raw: GNNInstance


def solve_oracle(instance_bundles, edge_caps, mem_caps, time_limit_s=30.0):
    if not CPSAT_AVAILABLE:
        raise RuntimeError("OR-Tools/CP-SAT is required: pip install ortools")
    result = CPSATAllocator(
        list(instance_bundles), edge_caps, mem_caps, time_limit_s=time_limit_s
    ).solve()
    selected = [(str(r), str(b)) for r, b in result.get("selected", [])]
    return (
        selected,
        float(result.get("total_utility", 0.0)),
        bool(result.get("is_optimal", False)),
        str(result.get("status", "UNKNOWN")),
    )


def topology_cases(full=False):
    cases = [
        (
            "chain8",
            lambda: generate_chain_topology(
                n_nodes=8, edge_capacity=4, memory_capacity=8,
                raw_fidelity=0.85, generation_prob=0.95, latency=5.0,
            ),
        ),
        (
            "grid3x3",
            lambda: generate_grid_topology(
                rows=3, cols=3, edge_capacity=4, memory_capacity=8,
                raw_fidelity=0.85, generation_prob=0.95, latency=5.0,
            ),
        ),
    ]
    if full:
        cases += [
            (
                "chain10",
                lambda: generate_chain_topology(
                    n_nodes=10, edge_capacity=5, memory_capacity=10,
                    raw_fidelity=0.88, generation_prob=0.90, latency=5.0,
                ),
            ),
            (
                "grid4x4",
                lambda: generate_grid_topology(
                    rows=4, cols=4, edge_capacity=5, memory_capacity=10,
                    raw_fidelity=0.88, generation_prob=0.90, latency=5.0,
                ),
            ),
        ]
    return cases


def build_dataset(
    split_name: str,
    seeds: Sequence[int],
    request_counts: Sequence[int],
    full_topologies: bool,
    oracle_time_limit_s: float,
    require_optimal: bool = True,
) -> List[GNNInstance]:
    out = []
    for topology_name, topology_fn in topology_cases(full_topologies):
        for seed in seeds:
            topology = topology_fn()
            generated = contention_sweep_instances(
                topology_fn, list(request_counts), seed=int(seed)
            )
            for workload_name, payload in generated.items():
                bundles = list(payload["bundles"])
                if not bundles:
                    continue
                sel, util, optimal, status = solve_oracle(
                    bundles,
                    payload["edge_capacities"],
                    payload["memory_capacities"],
                    time_limit_s=oracle_time_limit_s,
                )
                if require_optimal and not optimal:
                    print(
                        f"[{split_name}] skip {topology_name}/{workload_name}/"
                        f"seed={seed}: CP-SAT {status}"
                    )
                    continue
                iid = f"{split_name}:{topology_name}:{workload_name}:seed{seed}"
                out.append(
                    GNNInstance(
                        iid, topology_name, int(seed), topology, bundles,
                        dict(payload["edge_capacities"]),
                        dict(payload["memory_capacities"]),
                        sel, util, optimal, status,
                    )
                )
                print(
                    f"[{split_name}] {iid}: bundles={len(bundles)}, "
                    f"oracle_U={util:.3f}, status={status}"
                )
    if not out:
        raise RuntimeError(f"No usable {split_name} instances were generated")
    return out


def link_params(topology: dict, edge) -> dict:
    lp = topology.get("link_params", {})
    e = undirected(edge)
    return lp.get(e) or lp.get((e[1], e[0])) or {}


def tensorize(inst: GNNInstance, device: torch.device) -> TensorInstance:
    topo = inst.topology
    nodes = list(topo["nodes"])
    ni = {n: i for i, n in enumerate(nodes)}
    edges = [undirected(e) for e in topo["edges"]]

    degree = defaultdict(int)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    max_degree = max(list(degree.values()) + [1])
    max_mem = max([int(inst.memory_capacities.get(n, 0)) for n in nodes] + [1])
    max_cap = max([int(c) for c in inst.edge_capacities.values()] + [1])
    max_lat = max([float(link_params(topo, e).get("latency", 5.0)) for e in edges] + [1.0])

    node_rows = []
    for n in nodes:
        inc = [e for e in edges if n in e]
        caps = [
            float(inst.edge_capacities.get(e, inst.edge_capacities.get((e[1], e[0]), 0)))
            for e in inc
        ]
        fids = [float(link_params(topo, e).get("raw_fidelity", 0.85)) for e in inc]
        probs = [float(link_params(topo, e).get("generation_probability", 1.0)) for e in inc]
        lats = [float(link_params(topo, e).get("latency", 5.0)) for e in inc]
        node_rows.append([
            degree[n] / max_degree,
            float(inst.memory_capacities.get(n, 0)) / max_mem,
            float(np.mean(caps)) / max_cap if caps else 0.0,
            float(np.mean(fids)) if fids else 0.0,
            float(np.mean(probs)) if probs else 0.0,
            float(np.mean(lats)) / max_lat if lats else 0.0,
        ])
    node_x = torch.tensor(node_rows, dtype=torch.float32, device=device)

    srcs, dsts, edge_rows = [], [], []
    directed_id = {}
    for e in edges:
        u, v = e
        cap = float(inst.edge_capacities.get(e, inst.edge_capacities.get((v, u), 0)))
        lp = link_params(topo, e)
        feat = [
            cap / max_cap,
            float(lp.get("raw_fidelity", 0.85)),
            float(lp.get("generation_probability", 1.0)),
            float(lp.get("latency", 5.0)) / max_lat,
        ]
        for a, b in ((u, v), (v, u)):
            directed_id[(a, b)] = len(srcs)
            srcs.append(ni[a]); dsts.append(ni[b]); edge_rows.append(feat)

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_rows, dtype=torch.float32, device=device)

    max_u = max([abs(float(b.get("utility", 0.0))) for b in inst.bundles] + [1.0])
    max_bl = max([float(b.get("latency", 0.0)) for b in inst.bundles] + [1.0])
    max_cost = max([float(b.get("bell_pair_cost", 0.0)) for b in inst.bundles] + [1.0])
    total_ec = max(float(sum(inst.edge_capacities.values())), 1.0)
    total_mc = max(float(sum(inst.memory_capacities.values())), 1.0)
    n_nodes = max(len(nodes), 1)

    bx, paths, path_eids, keys, labels = [], [], [], [], []
    oracle = set(inst.oracle_selected)
    for b in inst.bundles:
        path = list(b.get("path", []))
        paths.append([ni[n] for n in path if n in ni])
        path_eids.append([
            directed_id[(u, v)] for u, v in zip(path, path[1:])
            if (u, v) in directed_id
        ])
        ed = {undirected(e): int(d) for e, d in b.get("edge_demands", {}).items()}
        md = {str(n): int(d) for n, d in b.get("memory_demands", {}).items()}
        er = []
        for e, d in ed.items():
            cap = float(inst.edge_capacities.get(e, inst.edge_capacities.get((e[1], e[0]), 0)))
            er.append(d / cap if cap > 0 else float(d))
        mr = []
        for n, d in md.items():
            cap = float(inst.memory_capacities.get(n, 0))
            mr.append(d / cap if cap > 0 else float(d))
        bx.append([
            float(b.get("utility", 0.0)) / max_u,
            float(b.get("fidelity", 0.0)),
            float(b.get("success_probability", 0.0)),
            float(b.get("latency", 0.0)) / max_bl,
            float(b.get("bell_pair_cost", 0.0)) / max_cost,
            max(len(path) - 1, 0) / n_nodes,
            sum(ed.values()) / total_ec,
            sum(md.values()) / total_mc,
            max(er, default=0.0),
            max(mr, default=0.0),
        ])
        k = bundle_key(b)
        keys.append(k)
        labels.append(1.0 if k in oracle else 0.0)

    return TensorInstance(
        node_x=node_x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        bundle_x=torch.tensor(bx, dtype=torch.float32, device=device),
        paths=paths,
        path_edge_ids=path_eids,
        keys=keys,
        labels=torch.tensor(labels, dtype=torch.float32, device=device),
        raw=inst,
    )


class EdgeAwareSAGE(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(hidden + EDGE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([h[src], edge_attr], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        deg = torch.zeros((h.size(0), 1), dtype=h.dtype, device=h.device)
        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), device=h.device))
        agg = agg / deg.clamp_min(1.0)
        z = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + self.dropout(z))


class QiOptGNN(nn.Module):
    def __init__(self, hidden_dim=64, layers=3, dropout=0.10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(NODE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(EDGE_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gnn = nn.ModuleList([
            EdgeAwareSAGE(hidden_dim, dropout) for _ in range(layers)
        ])
        head_in = hidden_dim * 5 + BUNDLE_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, item: TensorInstance):
        h = self.node_encoder(item.node_x)
        for layer in self.gnn:
            h = layer(h, item.edge_index, item.edge_attr)
        e = self.edge_encoder(item.edge_attr)
        global_h = h.mean(dim=0)

        reps = []
        for i, path in enumerate(item.paths):
            if path:
                pidx = torch.tensor(path, dtype=torch.long, device=h.device)
                path_h = h[pidx].mean(dim=0)
                src_h, dst_h = h[path[0]], h[path[-1]]
            else:
                path_h = src_h = dst_h = torch.zeros(self.hidden_dim, device=h.device)
            eids = item.path_edge_ids[i]
            if eids:
                eid = torch.tensor(eids, dtype=torch.long, device=h.device)
                path_e = e[eid].mean(dim=0)
            else:
                path_e = torch.zeros(self.hidden_dim, device=h.device)
            reps.append(torch.cat([
                path_h, src_h, dst_h, path_e, global_h, item.bundle_x[i]
            ], dim=-1))
        return self.head(torch.stack(reps, dim=0)).squeeze(-1)


def request_groups(keys: Sequence[BundleKey]):
    groups = defaultdict(list)
    for i, (rid, _) in enumerate(keys):
        groups[rid].append(i)
    return groups


def loss_fn(logits, labels, keys, pos_weight, pairwise_weight=0.25):
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if pairwise_weight <= 0:
        return bce
    pairs = []
    for inds in request_groups(keys).values():
        pos = [i for i in inds if labels[i].item() > 0.5]
        neg = [i for i in inds if labels[i].item() <= 0.5]
        if pos and neg:
            p = logits[pos[0]]
            n = logits[torch.tensor(neg, dtype=torch.long, device=logits.device)]
            pairs.append(F.softplus(-(p - n)).mean())
    return bce if not pairs else bce + pairwise_weight * torch.stack(pairs).mean()


def decode(inst: GNNInstance, scores: Dict[BundleKey, float], threshold=0.0):
    ranked = sorted(
        inst.bundles,
        key=lambda b: (scores.get(bundle_key(b), -math.inf), float(b.get("utility", 0.0))),
        reverse=True,
    )
    edge_load, mem_load = defaultdict(int), defaultdict(int)
    served, selected = set(), []
    for b in ranked:
        k = bundle_key(b); rid = k[0]
        if scores.get(k, -math.inf) < threshold:
            continue
        if float(b.get("utility", 0.0)) <= 0.0 or rid in served:
            continue
        fits = True
        for eraw, d in b.get("edge_demands", {}).items():
            e = undirected(eraw)
            cap = int(inst.edge_capacities.get(e, inst.edge_capacities.get((e[1], e[0]), 0)))
            if edge_load[e] + int(d) > cap:
                fits = False; break
        if fits:
            for n, d in b.get("memory_demands", {}).items():
                if mem_load[n] + int(d) > int(inst.memory_capacities.get(n, 0)):
                    fits = False; break
        if not fits:
            continue
        selected.append(k); served.add(rid)
        for eraw, d in b.get("edge_demands", {}).items():
            edge_load[undirected(eraw)] += int(d)
        for n, d in b.get("memory_demands", {}).items():
            mem_load[n] += int(d)
    return selected


def utility(selected, bundles):
    u = {bundle_key(b): float(b["utility"]) for b in bundles}
    return float(sum(u.get(k, 0.0) for k in selected))


def served_ratio(selected, bundles):
    reqs = {str(b["request_id"]) for b in bundles}
    return len({r for r, _ in selected}) / max(len(reqs), 1)


@torch.no_grad()
def scores_for(model, item):
    model.eval()
    logits = model(item).detach().cpu().tolist()
    return {k: float(v) for k, v in zip(item.keys, logits)}


@torch.no_grad()
def evaluate(model, items, threshold):
    rows = []
    for item in items:
        inst = item.raw
        selected = decode(inst, scores_for(model, item), threshold)
        gu = utility(selected, inst.bundles)
        gap = 100.0 * (inst.oracle_utility - gu) / max(abs(inst.oracle_utility), 1e-12)
        rows.append({
            "instance_id": inst.instance_id,
            "topology": inst.topology_name,
            "seed": inst.seed,
            "n_bundles": len(inst.bundles),
            "n_requests": len({b["request_id"] for b in inst.bundles}),
            "gnn_utility": gu,
            "oracle_utility": inst.oracle_utility,
            "optimality_gap_pct": gap,
            "gnn_served_ratio": served_ratio(selected, inst.bundles),
            "oracle_served_ratio": served_ratio(inst.oracle_selected, inst.bundles),
        })
    summary = {
        "n_instances": len(rows),
        "mean_utility": float(np.mean([r["gnn_utility"] for r in rows])),
        "mean_oracle_utility": float(np.mean([r["oracle_utility"] for r in rows])),
        "mean_optimality_gap_pct": float(np.mean([r["optimality_gap_pct"] for r in rows])),
        "median_optimality_gap_pct": float(np.median([r["optimality_gap_pct"] for r in rows])),
        "mean_served_ratio": float(np.mean([r["gnn_served_ratio"] for r in rows])),
    }
    return rows, summary


def tune_threshold(model, val_items):
    best_t, best_gap, best_summary = 0.0, math.inf, None
    for t in np.linspace(-3.0, 3.0, 61):
        _, s = evaluate(model, val_items, float(t))
        if s["mean_optimality_gap_pct"] < best_gap:
            best_t = float(t); best_gap = s["mean_optimality_gap_pct"]; best_summary = s
    return best_t, best_summary


def pos_weight(items, device):
    pos = sum(float(x.labels.sum().item()) for x in items)
    total = sum(int(x.labels.numel()) for x in items)
    neg = max(total - pos, 1.0); pos = max(pos, 1.0)
    return torch.tensor(min(neg / pos, 20.0), dtype=torch.float32, device=device)


def train(model, train_items, val_items, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pw = pos_weight(train_items, device)
    best_state, best_gap, best_epoch, best_t = None, math.inf, -1, 0.0
    patience_left, history = args.patience, []
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train(); order = list(range(len(train_items))); rng.shuffle(order); losses = []
        for idx in order:
            item = train_items[idx]
            logits = model(item)
            loss = loss_fn(logits, item.labels, item.keys, pw, args.pairwise_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(float(loss.detach().cpu()))
        t, val_summary = tune_threshold(model, val_items)
        gap = float(val_summary["mean_optimality_gap_pct"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_gap_pct": gap, "threshold": t})
        if gap < best_gap - 1e-9:
            best_gap, best_epoch, best_t = gap, epoch, t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:04d} loss={np.mean(losses):.5f} val_gap={gap:.3f}% threshold={t:+.2f}")
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    model.load_state_dict(best_state); model.to(device)
    return model, {"best_epoch": best_epoch, "best_val_gap_pct": best_gap, "threshold": best_t, "history": history}


def utility_density_greedy(inst: GNNInstance):
    scores = {}
    for b in inst.bundles:
        cost = sum(b.get("edge_demands", {}).values()) + sum(b.get("memory_demands", {}).values())
        scores[bundle_key(b)] = float(b["utility"]) / max(float(cost), 1e-9)
    return decode(inst, scores, threshold=-math.inf)


def evaluate_greedy(instances):
    gaps = []
    for inst in instances:
        u = utility(utility_density_greedy(inst), inst.bundles)
        gaps.append(100.0 * (inst.oracle_utility - u) / max(abs(inst.oracle_utility), 1e-12))
    return {"mean_optimality_gap_pct": float(np.mean(gaps)), "median_optimality_gap_pct": float(np.median(gaps))}


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description="QiOpt4QNet edge-aware GNN baseline")
    p.add_argument("--train-seeds", type=int, default=20)
    p.add_argument("--val-seeds", type=int, default=5)
    p.add_argument("--test-seeds", type=int, default=10)
    p.add_argument("--request-counts", type=int, nargs="+", default=[8, 16])
    p.add_argument("--full-topologies", action="store_true")
    p.add_argument("--oracle-time-limit", type=float, default=30.0)
    p.add_argument("--allow-feasible-labels", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pairwise-weight", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--out-dir", type=str, default="results/gnn_baseline")
    return p.parse_args()


def main():
    args = parse_args(); seed_everything(args.seed); device = resolve_device(args.device)
    print(f"Device: {device}")
    require_optimal = not args.allow_feasible_labels

    train_inst = build_dataset("train", range(args.train_seeds), args.request_counts, args.full_topologies, args.oracle_time_limit, require_optimal)
    val_inst = build_dataset("val", range(10000, 10000 + args.val_seeds), args.request_counts, args.full_topologies, args.oracle_time_limit, require_optimal)
    test_inst = build_dataset("test", range(20000, 20000 + args.test_seeds), args.request_counts, args.full_topologies, args.oracle_time_limit, require_optimal)

    train_items = [tensorize(x, device) for x in train_inst]
    val_items = [tensorize(x, device) for x in val_inst]
    test_items = [tensorize(x, device) for x in test_inst]

    model = QiOptGNN(args.hidden_dim, args.layers, args.dropout).to(device)
    print(f"GNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    t0 = time.perf_counter()
    model, report = train(model, train_items, val_items, args, device)
    train_time = time.perf_counter() - t0

    threshold = float(report["threshold"])
    test_rows, gnn_summary = evaluate(model, test_items, threshold)
    greedy_summary = evaluate_greedy(test_inst)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "gnn_test_instances.csv", test_rows)
    save_csv(out / "training_history.csv", report["history"])
    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
    }, out / "gnn_baseline.pt")

    summary = {
        "device": str(device),
        "train_instances": len(train_inst),
        "val_instances": len(val_inst),
        "test_instances": len(test_inst),
        "training_time_s": train_time,
        "best_epoch": report["best_epoch"],
        "decision_threshold": threshold,
        "gnn": gnn_summary,
        "utility_density_greedy": greedy_summary,
        "config": vars(args),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Held-out test ===")
    print(f"GNN mean gap:    {gnn_summary['mean_optimality_gap_pct']:.3f}%")
    print(f"GNN median gap:  {gnn_summary['median_optimality_gap_pct']:.3f}%")
    print(f"Greedy mean gap: {greedy_summary['mean_optimality_gap_pct']:.3f}%")
    print(f"Threshold:       {threshold:+.3f}")
    print(f"Outputs:         {out.resolve()}")


if __name__ == "__main__":
    main()