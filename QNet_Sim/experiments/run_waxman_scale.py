"""
Scale experiment: Waxman topologies with 20-50 nodes, 10-30 requests.
Compares all solvers on large, geographically distributed QKD networks.
"""
import sys, os, time, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from network.topology import generate_waxman
from network.request import Request
from routing.path_generator import PathGenerator
from routing.bundle_generation import BundleGenerator
from optimization.qubo_optimizer import QUBOOptimizer
from optimization.openjij_solver import solve_sa, solve_sqa
from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer
from optimization.refine_pipeline import TensorAnnealerPipeline
from optimization.baselines import (
    shortest_feasible_path,
    utility_density_greedy,
    fidelity_aware_greedy,
)


def _undirected_edge(e):
    return tuple(sorted(e))


def extract_capacities(net):
    edge_caps = {}
    for u, v, data in net.graph.edges(data=True):
        edge_caps[_undirected_edge((u, v))] = data["data"].capacity
    mem_caps = {}
    for nid, data in net.graph.nodes(data=True):
        mem_caps[nid] = data["data"].memory_capacity
    return edge_caps, mem_caps


def decode_best_feasible(optimizer, response, congestion_penalty=0.05,
                         memory_congestion_penalty=0.05):
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


def run_solver(name, bundles, edge_caps, mem_caps, method="qubo_sa", **kwargs):
    start = time.perf_counter()
    if method in ("qubo_sa", "qubo_sqa"):
        try:
            optimizer = QUBOOptimizer(bundles, edge_caps, mem_caps)
            bqm = optimizer.to_bqm(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                                   congestion_penalty=kwargs.get("congestion_penalty", 0.05),
                                   memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05))
            if method == "qubo_sqa":
                response = solve_sqa(bqm, num_reads=kwargs.get("sqa_num_reads", 10))
            else:
                response = solve_sa(bqm, num_reads=kwargs.get("num_reads", 20))
            selected, energy = decode_best_feasible(
                optimizer, response,
                kwargs.get("congestion_penalty", 0.05),
                kwargs.get("memory_congestion_penalty", 0.05))
        except Exception as e:
            return {"name": name, "selected": [], "energy": None, "time": time.perf_counter() - start, "error": str(e)}
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
            edge_penalty=10.0,
            memory_penalty=10.0,
            congestion_penalty=kwargs.get("congestion_penalty", 0.05),
            memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05),
        )
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": result.get("energy"), "time": elapsed}

    elif method == "pipeline":
        pipe = TensorAnnealerPipeline(bundles, edge_caps, mem_caps, seed=kwargs.get("seed", 42))
        result = pipe.solve(
            tn_bond_dim=kwargs.get("bond_dim", 8),
            tn_beta=kwargs.get("beta", 5.0),
            anneal_max_iterations=kwargs.get("max_iterations", 1500),
            anneal_initial_temperature=kwargs.get("initial_temperature", 3.0),
            congestion_penalty=kwargs.get("congestion_penalty", 0.05),
            memory_congestion_penalty=kwargs.get("memory_congestion_penalty", 0.05),
        )
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": result["final_energy"], "time": elapsed}

    elif method == "utility_density_greedy":
        result = utility_density_greedy(bundles, edge_caps, mem_caps)
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": None, "time": elapsed}

    elif method == "fidelity_aware_greedy":
        result = fidelity_aware_greedy(bundles, edge_caps, mem_caps)
        elapsed = time.perf_counter() - start
        return {"name": name, "selected": result["selected"], "energy": None, "time": elapsed}

    return {"name": name, "selected": [], "energy": None, "time": 0}


def compute_metrics(result, bundles, edge_caps, mem_caps):
    selected_map = {rid: bid for rid, bid in result.get("selected", [])}
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
        if load > edge_caps.get(e, 0):
            violations += 1
    for n, load in mem_load.items():
        if load > mem_caps.get(n, 0):
            violations += 1
    return {"served": served, "total_utility": total_utility, "violations": violations}


def main():
    random.seed(12345)
    all_results = []

    solvers = [
        ("QUBO+SA (OpenJij)", "qubo_sa", {"num_reads": 20}),
        ("QUBO+SQA (OpenJij)", "qubo_sqa", {"sqa_num_reads": 10}),
        ("Metropolis Annealer", "metropolis", {}),
        ("Tensor Network MPS", "tensor_network", {}),
        ("TN+Metro Pipeline", "pipeline", {}),
        ("Utility-Density Greedy", "utility_density_greedy", {}),
        ("Fidelity-Aware Greedy", "fidelity_aware_greedy", {}),
    ]

    waxman_configs = [
        (20, 10, 0.4, 0.4),
        (30, 15, 0.3, 0.3),
        (40, 20, 0.25, 0.25),
        (50, 30, 0.2, 0.2),
    ]

    for n_nodes, n_req, alpha, beta in waxman_configs:
        label = f"waxman_{n_nodes}_req_{n_req}"
        print(f"\n--- {label} ---")
        net = generate_waxman(n_nodes, alpha=alpha, beta=beta,
                              default_memory=(5, 15),
                              link_params={
                                  "capacity_bounds": (2, 6),
                                  "fidelity_bounds": (0.8, 0.95),
                              })
        ec, mc = extract_capacities(net)
        print(f"  Nodes: {net.graph.number_of_nodes()}, Edges: {net.graph.number_of_edges()}")

        candidates = [n for n in net.graph.nodes() if net.graph.degree(n) >= 1]
        if len(candidates) < 2:
            print("  Skipping: too few connected nodes")
            continue

        reqs = []
        for i in range(n_req):
            src, dst = random.sample(candidates, 2)
            reqs.append(Request(src, dst, random.uniform(0.5, 0.75),
                                weight=random.uniform(10.0, 100.0),
                                request_id=f"Req_{i}"))

        gen = BundleGenerator(net)
        path_gen = PathGenerator(net)
        all_bundles = []
        for req in reqs:
            paths = path_gen.get_k_shortest_paths(req, k=3)
            if not paths:
                continue
            bundles = gen.generate_bundles(req, paths)
            for b in bundles:
                all_bundles.append(b.to_optimizer_dict())

        if not all_bundles:
            print("  Skipping: no viable bundles")
            continue

        print(f"  Requests: {len(reqs)}, Bundles: {len(all_bundles)}")

        row = {"label": label, "n_nodes": n_nodes, "n_requests": n_req, "n_bundles": len(all_bundles), "results": []}
        for sname, smethod, skwargs in solvers:
            result = run_solver(sname, all_bundles, ec, mc, method=smethod, **skwargs)
            metrics = compute_metrics(result, all_bundles, ec, mc)
            result.update(metrics)
            row["results"].append(result)
            en_str = f"{result.get('energy', 'N/A'):.1f}" if result.get('energy') is not None else "N/A"
            print(f"  {sname:<30} served={metrics['served']:>2}  utility={metrics['total_utility']:>8.1f}  "
                  f"viols={metrics['violations']:>2}  time={result['time']:>7.3f}s  energy={en_str}")

        all_results.append(row)

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "waxman_scale_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
