"""
Research experiment runner for QiOpt4QNet optimizer comparison.

Runs all optimizers and baselines across chain/grid topologies
with varying request loads, and reports:
  - served-request ratio
  - aggregate utility
  - runtime
  - scalability metrics

Usage:
    python experiments/run_comparison.py
"""
import sys, os, time, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import networkx as nx
from network.network import QuantumNetwork
from network.node import QuantumNode
from network.link import QuantumLink
from network.request import Request
from network.topology import generate_chain, generate_grid, generate_random
from routing.path_generator import PathGenerator
from routing.bundle_generation import BundleGenerator
from optimization.qubo_optimizer import QUBOOptimizer
from optimization.openjij_solver import solve_sa, solve_sqa
from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
from optimization.sequential_branch_optimizer import SequentialBranchOptimizer
from optimization.baselines import (
    shortest_feasible_path,
    utility_density_greedy,
    fidelity_aware_greedy,
)


def _undirected_edge(e):
    return tuple(sorted(e))


def generate_random_requests(net, num_requests, seed=None):
    rng = random.Random(seed)
    nodes = list(net.graph.nodes())
    reqs = []
    # Filter: only consider nodes with degree >= 1 (not isolated)
    candidates = [n for n in nodes if net.graph.degree(n) >= 1]
    if len(candidates) < 2:
        return []
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


def run_solver(name, bundles, edge_caps, mem_caps, method="qubo_sa", **kwargs):
    start = time.perf_counter()
    if method in ("qubo_sa", "qubo_sqa"):
        optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)
        bqm = optimizer.to_bqm(
                               congestion_penalty=kwargs.get("congestion_penalty", 0.05),
                               memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05))
        if method == "qubo_sqa":
            response = solve_sqa(bqm, num_reads=kwargs.get("sqa_num_reads", 10))
        else:
            response = solve_sa(bqm, num_reads=kwargs.get("num_reads", 50))
        selected, energy = decode_best_feasible(
            optimizer, response,
            kwargs.get("congestion_penalty", 0.05),
            kwargs.get("memory_congestion_penalty", 0.05))
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": selected, "energy": energy, "time": elapsed}

    elif method == "metropolis":
        opt = MetropolisAnnealer(bundles, edge_caps, mem_caps, seed=kwargs.get("seed", 42))
        result = opt.solve(
            max_iterations=kwargs.get("max_iterations", 3000),
            initial_temperature=kwargs.get("initial_temperature", 10.0),
            cooling_rate=kwargs.get("cooling_rate", 0.97),
            congestion_penalty=kwargs.get("congestion_penalty", 0.05),
            memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05),
        )
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": result["energy"], "time": elapsed}

    elif method == "tensor_network":
        opt = TensorNetworkOptimizer(bundles, edge_caps, mem_caps)
        result = opt.solve(
            bond_dim=kwargs.get("bond_dim", 8),
            beta=kwargs.get("beta", 5.0),
            congestion_penalty=kwargs.get("congestion_penalty", 0.05),
            memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05),
        )
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": result.get("energy"), "time": elapsed}

    elif method == "sequential_branch":
        opt = SequentialBranchOptimizer(bundles, edge_caps, mem_caps)
        result = opt.solve(
            branch_factor=kwargs.get("branch_factor", 8),
            beta=kwargs.get("beta", 5.0),
            congestion_penalty=kwargs.get("congestion_penalty", 0.05),
            memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05),
        )
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": result["energy"], "time": elapsed}

    elif method == "utility_density_greedy":
        result = utility_density_greedy(bundles, edge_caps, mem_caps)
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": None, "time": elapsed}

    elif method == "fidelity_aware_greedy":
        result = fidelity_aware_greedy(bundles, edge_caps, mem_caps)
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": None, "time": elapsed}

    return {"name": name, "selected": [], "energy": None, "time": 0}


def decode_best_feasible(optimizer, response, congestion_penalty=0.05,
                         memory_congestion_penalty=0.05):
    """Decode and repair every sample, then return the best-energy feasible one."""
    best = []
    best_energy = float("inf")
    for sample in response.samples():
        sel = optimizer.decode_sample(sample, repair=True)
        if not sel:
            continue
        en = optimizer.solution_energy(
            sel,
            congestion_penalty=congestion_penalty,
            memory_congestion_penalty=memory_congestion_penalty,
        )
        if en < best_energy:
            best_energy = en
            best = sel
    return best, best_energy


def compute_metrics(result, bundles, edge_caps, mem_caps):
    selected_map = {rid: bid for rid, bid in result["selected"]}
    served = len(selected_map)
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
        cap = edge_caps.get(e, 0)
        if load > cap:
            violations += 1

    for n, load in mem_load.items():
        cap = mem_caps.get(n, 0)
        if load > cap:
            violations += 1

    return {
        "served": served,
        "total_utility": total_utility,
        "violations": violations,
    }


def run_experiment(net, requests, edge_caps, mem_caps, label=""):
    gen = BundleGenerator(net)
    path_gen = PathGenerator(net)

    all_bundles = []
    for req in requests:
        paths = path_gen.get_k_shortest_paths(req, k=3)
        if not paths:
            continue
        bundles = gen.generate_bundles(req, paths)
        for b in bundles:
            all_bundles.append(b.to_optimizer_dict())

    if not all_bundles:
        print(f"  [{label}] No viable bundles generated")
        return None

    solvers = [
        ("QUBO+SA (OpenJij)", "qubo_sa"),
        ("QUBO+SQA (OpenJij)", "qubo_sqa"),
        ("Metropolis Annealer", "metropolis"),
        ("Sequential Branch Expansion", "sequential_branch"),
        ("Tensor Network MPS", "tensor_network"),
        ("Utility-Density Greedy", "utility_density_greedy"),
        ("Fidelity-Aware Greedy", "fidelity_aware_greedy"),
    ]

    results = []
    for sname, smethod in solvers:
        result = run_solver(sname, all_bundles, edge_caps, mem_caps, method=smethod)
        metrics = compute_metrics(result, all_bundles, edge_caps, mem_caps)
        result.update(metrics)
        results.append(result)

    return {
        "label": label,
        "num_requests": len(requests),
        "num_bundles": len(all_bundles),
        "results": results,
    }


def print_results(exp):
    print(f"\n{'='*70}")
    print(f"  Experiment: {exp['label']}")
    print(f"  Requests: {exp['num_requests']},  Bundles: {exp['num_bundles']}")
    print(f"{'='*70}")
    header = f"{'Solver':<30} {'Served':>8} {'Utility':>12} {'Viols':>6} {'Time (s)':>10} {'Energy':>12}"
    print(header)
    print("-" * len(header))
    for r in exp["results"]:
        energy_str = f"{r['energy']:<12.4f}" if r["energy"] is not None else f"{'N/A':>12}"
        print(
            f"{r['name']:<30} {r['served']:>8} {r['total_utility']:>12.2f} "
            f"{r['violations']:>6} {r['time']:>10.4f} {energy_str}"
        )


def main():
    random.seed(12345)
    all_experiments = []

    print("=" * 70)
    print("  QiOpt4QNet Optimizer Research Comparison")
    print("  arXiv:2605.27425 methods vs classical baselines")
    print("=" * 70)

    # --- Chain topologies ---
    for n_nodes in [4, 6, 8]:
        print(f"\n--- Chain: {n_nodes} nodes ---")
        net = generate_chain(n_nodes, default_memory=(5, 15),
                             link_params={"capacity_bounds": (2, 6), "fidelity_bounds": (0.8, 0.95)})
        ec, mc = extract_capacities(net)
        for n_req in [2, 4, 6, 8]:
            reqs = generate_random_requests(net, n_req, seed=random.randint(0, 10000))
            if len(reqs) < 2:
                continue
            exp = run_experiment(net, reqs, ec, mc, label=f"chain_{n_nodes}_req_{n_req}")
            if exp:
                all_experiments.append(exp)
                print_results(exp)

    # --- Grid topologies ---
    for m, n in [(3, 3), (3, 4), (4, 4)]:
        print(f"\n--- Grid: {m}x{n} ---")
        net = generate_grid(m, n, default_memory=(5, 15),
                            link_params={"capacity_bounds": (2, 6), "fidelity_bounds": (0.8, 0.95)})
        ec, mc = extract_capacities(net)
        for n_req in [3, 6, 10]:
            reqs = generate_random_requests(net, n_req, seed=random.randint(0, 10000))
            if len(reqs) < 2:
                continue
            exp = run_experiment(net, reqs, ec, mc, label=f"grid_{m}x{n}_req_{n_req}")
            if exp:
                all_experiments.append(exp)
                print_results(exp)

    # Write summary JSON
    summary_path = os.path.join(os.path.dirname(__file__), "..", "results", "experiment_results.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_experiments, f, indent=2, default=str)
    print(f"\nResults written to {summary_path}")


if __name__ == "__main__":
    main()
