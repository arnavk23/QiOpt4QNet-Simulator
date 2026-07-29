"""
Q3: Can Tensor Network pre-screening reduce the Metropolis Annealer's search space?

Pipeline: TN (low χ) → filter bundles → Metropolis on reduced set.
Compare: Metro-only vs TN→Metro pipeline on same instances.
"""
import sys, os, time, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from network.topology import generate_grid, generate_chain
from network.request import Request
from routing.path_generator import PathGenerator
from routing.bundle_generation import BundleGenerator
from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer


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


def run_pipeline(net, reqs, ec, mc, label=""):
    path_gen = PathGenerator(net)
    gen = BundleGenerator(net)

    all_bundles = []
    for req in reqs:
        paths = path_gen.get_k_shortest_paths(req, k=3)
        if not paths:
            continue
        bundles = gen.generate_bundles(req, paths)
        for b in bundles:
            all_bundles.append(b.to_optimizer_dict())

    if not all_bundles:
        return None

    # 1. Metro-only (baseline)
    t0 = time.perf_counter()
    metro = MetropolisAnnealer(all_bundles, ec, mc, seed=42)
    mresult = metro.solve(max_iterations=3000)
    mtime = time.perf_counter() - t0
    mmetrics = compute_metrics(mresult["selected"], all_bundles, ec, mc)

    # 2. TN pre-screen (low χ) → filter → Metro
    #    Keep only bundles selected by TN for each request (or best 2 if TN selected none)
    t0 = time.perf_counter()
    tn = TensorNetworkOptimizer(all_bundles, ec, mc)
    tresult = tn.solve(bond_dim=4, beta=5.0)
    tn_selected = {rid: bid for rid, bid in tresult["selected"]}

    # Filter bundles to TN-selected ones (plus at most 1 alternative per request)
    filtered = []
    request_counts = {}
    for b in all_bundles:
        rid = b["request_id"]
        if rid in tn_selected and b["bundle_id"] == tn_selected[rid]:
            filtered.append(b)
            request_counts[rid] = request_counts.get(rid, 0) + 1
        elif rid not in tn_selected or rid not in request_counts:
            filtered.append(b)
            request_counts[rid] = request_counts.get(rid, 0) + 1

    # Add fallback: if any request has no bundles in filtered, add its best bundle
    for b in all_bundles:
        rid = b["request_id"]
        if request_counts.get(rid, 0) == 0:
            filtered.append(b)
            request_counts[rid] = 1

    tn_filter_time = time.perf_counter() - t0

    # Metro on filtered set
    t0 = time.perf_counter()
    metro2 = MetropolisAnnealer(filtered, ec, mc, seed=42)
    m2result = metro2.solve(max_iterations=3000)
    m2time = time.perf_counter() - t0
    m2metrics = compute_metrics(m2result["selected"], all_bundles, ec, mc)

    total_time = tn_filter_time + m2time

    return {
        "label": label,
        "num_requests": len(reqs),
        "total_bundles": len(all_bundles),
        "filtered_bundles": len(filtered),
        "reduction_pct": (1 - len(filtered) / max(len(all_bundles), 1)) * 100,
        "metro_only": {"utility": mmetrics["utility"], "served": mmetrics["served"],
                        "violations": mmetrics["violations"], "time": mtime},
        "tn_metro": {"utility": m2metrics["utility"], "served": m2metrics["served"],
                      "violations": m2metrics["violations"], "time": total_time,
                      "tn_time": tn_filter_time, "metro_time": m2time},
    }


def main():
    random.seed(42)
    print("=" * 70)
    print("  Q3: Tensor Network Pre-Screening for Metropolis Annealer")
    print("=" * 70)

    all_results = []

    for topo_name, topo_fn, params in [
        ("chain_6", lambda: generate_chain(6, default_memory=(5, 15),
                    link_params={"capacity_bounds": (2, 6), "fidelity_bounds": (0.8, 0.95)}), [3, 5, 8]),
        ("grid_3x4", lambda: generate_grid(3, 4, default_memory=(5, 15),
                    link_params={"capacity_bounds": (2, 6), "fidelity_bounds": (0.8, 0.95)}), [3, 5, 8]),
        ("grid_4x4", lambda: generate_grid(4, 4, default_memory=(5, 15),
                    link_params={"capacity_bounds": (2, 6), "fidelity_bounds": (0.8, 0.95)}), [5, 10]),
    ]:
        net = topo_fn()
        ec, mc = extract_capacities(net)

        for n_req in params:
            reqs = generate_random_requests(net, n_req, seed=random.randint(0, 10000))
            if len(reqs) < 2:
                continue
            result = run_pipeline(net, reqs, ec, mc, label=f"{topo_name}_req_{n_req}")
            if result:
                all_results.append(result)
                r = result
                print(f"\n--- {r['label']} ---")
                print(f"  Bundles: {r['total_bundles']} -> {r['filtered_bundles']} ({r['reduction_pct']:.0f}% reduction)")
                print(f"  Metro alone:     U={r['metro_only']['utility']:.1f}, "
                      f"served={r['metro_only']['served']}, viol={r['metro_only']['violations']}, "
                      f"t={r['metro_only']['time']:.4f}s")
                print(f"  TN→Metro pipe:   U={r['tn_metro']['utility']:.1f}, "
                      f"served={r['tn_metro']['served']}, viol={r['tn_metro']['violations']}, "
                      f"t={r['tn_metro']['time']:.4f}s "
                      f"(TN={r['tn_metro']['tn_time']:.4f}s + Metro={r['tn_metro']['metro_time']:.4f}s)")

    report_path = os.path.join(os.path.dirname(__file__), "..", "results", "tensor_prescreening.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {report_path}")


if __name__ == "__main__":
    main()
