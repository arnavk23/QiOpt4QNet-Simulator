import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, 'src')))

from network import topology
from routing.bundle_generation import BundleGenerator
from optimization.metropolis_annealer import MetropolisAnnealer

from baselines.request_generator import generate_requests
from baselines.path_strategies import CandidatePathGenerator
from baselines.classical_baselines import run_all_baselines, CPSAT_AVAILABLE
from baselines.feasibility import compare_all, print_comparison


def main():
    print("=" * 70)
    print("  CLASSICAL BASELINES vs QUANTUM OPTIMIZERS — QiOpt4QNet")
    print("=" * 70)

    # 1. Network + workload generation (grid topology, medium load)
    net = topology.generate_grid(3, 3, default_memory=8, link_params={
        "fidelity_bounds": (0.75, 0.95),
        "prob_bounds": (0.5, 0.9),
        "capacity_bounds": (2, 6),
    })
    requests = generate_requests(net, load="medium", seed=7)
    print(f"\nNetwork: {net.graph.number_of_nodes()} nodes, {net.graph.number_of_edges()} edges")
    print(f"Requests: {len(requests)}")

    # 2. Candidate paths + bundles (multiple path strategies, multiple purification levels)
    path_gen = CandidatePathGenerator(net)
    bundle_gen = BundleGenerator(net)
    bundles = []
    for req in requests:
        candidate_paths = path_gen.generate_candidates(req, k=3)
        bundles.extend(bundle_gen.generate_bundles(req, candidate_paths))
    print(f"Candidate bundles generated: {len(bundles)}")

    edge_caps = {tuple(sorted((u, v))): d["data"].capacity for u, v, d in net.graph.edges(data=True)}
    mem_caps = {n: d["data"].memory_capacity for n, d in net.graph.nodes(data=True)}
    opt_bundles = [b.to_optimizer_dict() for b in bundles]

    # 3. Run every classical baseline
    print(f"\n---> Running classical baselines (CP-SAT {'available' if CPSAT_AVAILABLE else 'NOT available'}) <---")
    results = run_all_baselines(opt_bundles, edge_caps, mem_caps, seed=42)

    # 4. Run the existing quantum/annealing optimizer for comparison
    metro = MetropolisAnnealer(opt_bundles, edge_caps, mem_caps, seed=42)
    results["metropolis_annealer"] = metro.solve(max_iterations=3000)
    results["metropolis_annealer"]["method"] = "metropolis_annealer"

    # 5. Common evaluation
    all_rids = [r.request_id for r in requests]
    metrics = compare_all(results, opt_bundles, edge_caps, mem_caps, all_request_ids=all_rids)

    print()
    print_comparison(metrics)


if __name__ == "__main__":
    main()
