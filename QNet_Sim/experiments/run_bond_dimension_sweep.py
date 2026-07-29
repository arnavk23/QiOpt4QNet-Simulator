"""
Q2: How does bond dimension χ affect Tensor Network MPS accuracy and cost?

Tests χ ∈ {1, 2, 4, 8, 16} on a fixed grid topology with varying request counts.
"""
import sys, os, time, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from network.topology import generate_grid
from network.request import Request
from routing.path_generator import PathGenerator
from routing.bundle_generation import BundleGenerator
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
from optimization.metropolis_annealer import MetropolisAnnealer


def _undirected_edge(e):
    return tuple(sorted(e))


def generate_random_requests(net, num_requests, seed=None):
    rng = random.Random(seed)
    nodes = list(net.graph.nodes())
    candidates = [n for n in nodes if net.graph.degree(n) >= 1]
    reqs = []
    for i in range(num_requests):
        src, dst = rng.sample(candidates, 2)
        w = rng.uniform(10.0, 100.0)
        min_fid = rng.uniform(0.5, 0.75)
        reqs.append(Request(src, dst, min_fid, weight=w, request_id=f"Req_{i}"))
    return reqs


def extract_capacities(net):
    edge_caps = {}
    for u, v, data in net.graph.edges(data=True):
        edge_caps[_undirected_edge((u, v))] = data["data"].capacity
    mem_caps = {}
    for nid, data in net.graph.nodes(data=True):
        mem_caps[nid] = data["data"].memory_capacity
    return edge_caps, mem_caps


def compute_metrics(selected_pairs, bundles, edge_caps, mem_caps):
    selected_map = {rid: bid for rid, bid in selected_pairs}
    total_utility = 0.0
    edge_load = {}
    mem_load = {}
    violations = 0
    for b in bundles:
        rid = b["request_id"]
        bid = b["bundle_id"]
        if rid in selected_map and selected_map[rid] == bid:
            total_utility += b["utility"]
            for e, d in b["edge_demands"].items():
                eu = _undirected_edge(e)
                edge_load[eu] = edge_load.get(eu, 0) + d
            for n, d in b["memory_demands"].items():
                mem_load[n] = mem_load.get(n, 0) + d
    for e, load in edge_load.items():
        if load > edge_caps.get(e, 0):
            violations += 1
    for n, load in mem_load.items():
        if load > mem_caps.get(n, 0):
            violations += 1
    return {"utility": total_utility, "violations": violations, "served": len(selected_map)}


def main():
    random.seed(42)
    print("=" * 70)
    print("  Q2: Bond Dimension χ Sweep for Tensor Network MPS")
    print("=" * 70)

    bond_dims = [1, 2, 4, 8, 16]
    net = generate_grid(3, 4, default_memory=(5, 15),
                        link_params={"capacity_bounds": (2, 6), "fidelity_bounds": (0.8, 0.95)})
    ec, mc = extract_capacities(net)
    path_gen = PathGenerator(net)
    gen = BundleGenerator(net)

    for n_req in [3, 5, 8]:
        print(f"\n--- {n_req} requests on 3x4 grid ---")
        reqs = generate_random_requests(net, n_req, seed=42 + n_req)
        all_bundles = []
        for req in reqs:
            paths = path_gen.get_k_shortest_paths(req, k=3)
            if not paths:
                continue
            bundles = gen.generate_bundles(req, paths)
            for b in bundles:
                all_bundles.append(b.to_optimizer_dict())

        if not all_bundles:
            continue

        # Metropolis reference
        start = time.perf_counter()
        metro = MetropolisAnnealer(all_bundles, ec, mc, seed=42)
        mresult = metro.solve(max_iterations=3000)
        mtime = time.perf_counter() - start
        mmetrics = compute_metrics(mresult["selected"], all_bundles, ec, mc)

        print(f"{'χ':>5} {'Served':>8} {'Utility':>12} {'Viols':>6} {'Time (s)':>10} {'vs Metro %':>12}")
        print("-" * 55)
        print(f"{'Metro':>5} {mmetrics['served']:>8} {mmetrics['utility']:>12.2f} "
              f"{mmetrics['violations']:>6} {mtime:>10.4f} {'100.0%':>12}")

        for chi in bond_dims:
            start = time.perf_counter()
            tn = TensorNetworkOptimizer(all_bundles, ec, mc)
            tresult = tn.solve(bond_dim=chi, beta=5.0)
            ttime = time.perf_counter() - start
            tmetrics = compute_metrics(tresult["selected"], all_bundles, ec, mc)
            pct = (tmetrics["utility"] / mmetrics["utility"] * 100) if mmetrics["utility"] > 0 else 0
            print(f"{chi:>5} {tmetrics['served']:>8} {tmetrics['utility']:>12.2f} "
                  f"{tmetrics['violations']:>6} {ttime:>10.4f} {pct:>11.1f}%")


if __name__ == "__main__":
    main()
