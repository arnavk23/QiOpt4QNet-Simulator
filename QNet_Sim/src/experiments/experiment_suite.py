import sys, os, csv, math, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dimod

from experiments.instances import (
    generate_chain_topology, generate_grid_topology,
    contention_sweep_instances,
)
from experiments.metrics import ExperimentTracker
from experiments.benchmark import build_metropolis, build_tensor_network
from optimization.physics_hamiltonian import PhysicalHamiltonian

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiments"))
os.makedirs(OUT, exist_ok=True)


def run_contention_sweep():
    """Vary request count on chain topology at different edge capacities."""
    all_rows = []
    for cap in [4, 6, 10]:
        for n_nodes in [4, 6, 10]:
            topo_fn = lambda n=n_nodes, c=cap: generate_chain_topology(
                n_nodes=n, edge_capacity=c, latency=5.0, raw_fidelity=0.85
            )
            instances = contention_sweep_instances(topo_fn, [2, 4, 8, 16, 24], seed=42)
            for name, inst in instances.items():
                n_req = inst["n_requests"]
                n_bun = len(inst["bundles"])
                b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

                for seed in [42, 43, 44]:
                    meta_sf = build_metropolis(b, ec, mc, seed=seed)
                    t0 = time.perf_counter()
                    meta_r = meta_sf(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                                     congestion_penalty=0.0, memory_congestion_penalty=0.0,
                                     max_iterations=5000, cooling_rate=0.97,
                                     n_restarts=1, steps_per_temperature=10)
                    mt = time.perf_counter() - t0

                    tn_sf = build_tensor_network(b, ec, mc)
                    t0 = time.perf_counter()
                    tn_r = tn_sf(bond_dim=8, beta=5.0, edge_penalty=10.0, memory_penalty=10.0)
                    tt = time.perf_counter() - t0

                    for solver, r, t in [("Metropolis", meta_r, mt), ("TensorNetwork", tn_r, tt)]:
                        all_rows.append({
                            "topology": f"chain_{n_nodes}",
                            "n_nodes": n_nodes,
                            "edge_capacity": cap,
                            "n_requests": n_req,
                            "n_bundles": n_bun,
                            "solver": solver,
                            "seed": seed,
                            "served": len(r.get("selected", [])),
                            "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                            "utility": sum(_u(b, rid, bid) for rid, bid in r.get("selected", [])),
                            "time_s": t,
                        })

    path = os.path.join(OUT, "contention_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")
    return all_rows


def _u(bundles, rid, bid):
    for b in bundles:
        if b["request_id"] == rid and b["bundle_id"] == bid:
            return b["utility"]
    return 0.0


def run_bond_dim_sweep():
    """Vary TN bond dimension across request counts."""
    all_rows = []
    topo_fn = lambda: generate_chain_topology(n_nodes=8, edge_capacity=6)
    instances = contention_sweep_instances(topo_fn, [4, 8, 16, 24], seed=42)

    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

        for bd in [1, 2, 4, 8, 16, 32]:
            tn_sf = build_tensor_network(b, ec, mc)
            t0 = time.perf_counter()
            tn_r = tn_sf(bond_dim=bd, beta=5.0, edge_penalty=10.0, memory_penalty=10.0)
            tt = time.perf_counter() - t0
            all_rows.append({
                "topology": "chain_8",
                "n_requests": n_req,
                "bond_dim": bd,
                "solver": f"TN(χ={bd})",
                "served": len(tn_r.get("selected", [])),
                "served_ratio": len(tn_r.get("selected", [])) / max(n_req, 1),
                "utility": sum(_u(b, rid, bid) for rid, bid in tn_r.get("selected", [])),
                "time_s": tt,
            })

        meta_sf = build_metropolis(b, ec, mc, seed=42)
        t0 = time.perf_counter()
        meta_r = meta_sf(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                         congestion_penalty=0.0, memory_congestion_penalty=0.0,
                         max_iterations=5000, cooling_rate=0.97,
                         n_restarts=1, steps_per_temperature=10)
        mt = time.perf_counter() - t0
        all_rows.append({
            "topology": "chain_8",
            "n_requests": n_req,
            "bond_dim": -1,
            "solver": "Metropolis-SA",
            "served": len(meta_r.get("selected", [])),
            "served_ratio": len(meta_r.get("selected", [])) / max(n_req, 1),
            "utility": sum(_u(b, rid, bid) for rid, bid in meta_r.get("selected", [])),
            "time_s": mt,
        })

    path = os.path.join(OUT, "bond_dim_sweep.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_grid_comparison():
    """Compare solvers on grid topology."""
    all_rows = []
    for rows, cols in [(2, 3), (3, 3), (4, 4)]:
        topo_fn = lambda r=rows, c=cols: generate_grid_topology(
            rows=r, cols=c, edge_capacity=5, latency=5.0
        )
        instances = contention_sweep_instances(topo_fn, [8, 16], seed=42)
        for name, inst in instances.items():
            n_req = inst["n_requests"]
            b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

            meta_sf = build_metropolis(b, ec, mc, seed=42)
            t0 = time.perf_counter()
            meta_r = meta_sf(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                             congestion_penalty=0.0, memory_congestion_penalty=0.0,
                             max_iterations=5000, cooling_rate=0.97,
                             n_restarts=1, steps_per_temperature=10)
            mt = time.perf_counter() - t0

            tn_sf = build_tensor_network(b, ec, mc)
            t0 = time.perf_counter()
            tn_r = tn_sf(bond_dim=8, beta=5.0, edge_penalty=10.0, memory_penalty=10.0)
            tt = time.perf_counter() - t0

            for solver, r, t in [("Metropolis", meta_r, mt), ("TensorNetwork", tn_r, tt)]:
                all_rows.append({
                    "topology": f"grid_{rows}x{cols}",
                    "n_requests": n_req,
                    "solver": solver,
                    "served": len(r.get("selected", [])),
                    "served_ratio": len(r.get("selected", [])) / max(n_req, 1),
                    "utility": sum(_u(b, rid, bid) for rid, bid in r.get("selected", [])),
                    "time_s": t,
                })

    path = os.path.join(OUT, "grid_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_streaming_comparison():
    """Compare streaming vs batched on a request arrival sequence."""
    from optimization.streaming_annealer import StreamingAnnealer

    all_rows = []
    for n_req in [4, 8, 12, 16]:
        topo = generate_chain_topology(n_nodes=8, edge_capacity=6)
        pairs = []
        rng = __import__("random").Random(42)
        nodes = topo["nodes"]
        for _ in range(n_req):
            src, dst = rng.sample(nodes, 2)
            pairs.append((src, dst, rng.uniform(10, 100), rng.uniform(0.5, 0.8)))

        from experiments.instances import generate_request_bundles
        all_bundles = []
        all_by_rid = {}
        for i, (src, dst, w, mf) in enumerate(pairs):
            rid = f"req_{i}"
            bs = generate_request_bundles(topo, src, dst, w, mf)
            for j, b in enumerate(bs):
                b["bundle_id"] = f"{rid}_b{j}"
                b["request_id"] = rid
            all_by_rid[rid] = bs
            all_bundles.extend(bs)

        # Streaming: add incrementally
        sa = StreamingAnnealer(topo["edge_capacities"], topo["memory_capacities"], seed=42)
        t0 = time.perf_counter()
        for rid, bs in all_by_rid.items():
            sa.add_request(rid, bs)
            sa.local_sweep(n_steps=50, temperature=2.0)
        st = time.perf_counter() - t0
        sa_sel = sa.get_selected()

        # Batched: solve from scratch
        opt = __import__("optimization.metropolis_annealer", fromlist=["MetropolisAnnealer"]).MetropolisAnnealer(
            all_bundles, topo["edge_capacities"], topo["memory_capacities"], seed=42
        )
        t0 = time.perf_counter()
        r = opt.solve(penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
                      max_iterations=3000, n_restarts=1, steps_per_temperature=10)
        bt = time.perf_counter() - t0

        for solver, sel, t in [("Streaming", sa_sel, st), ("Batched", r["selected"], bt)]:
            all_rows.append({
                "n_requests": n_req,
                "solver": solver,
                "served": len(sel),
                "served_ratio": len(sel) / max(n_req, 1),
                "utility": sum(_u(all_bundles, rid, bid) for rid, bid in sel),
                "time_s": t,
            })

    path = os.path.join(OUT, "streaming_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")


def run_hamiltonian_encoding_comparison():
    """Extension 12.1: slack-variable QUBO vs direct (slack-free) Hamiltonian.

    Both formulations are compiled through pyqubo and solved with OpenJij SA on
    identical instances, so the only difference is the constraint encoding:
    slack variables + squared-deviation terms (Eq. 5.3-5.4) vs the direct
    max(0, .)^2 capacity terms with fixed-scale lambdas (Eq. 12.1).  We sweep
    the penalty/constraint scale to quantify penalty sensitivity, the main
    weakness of the slack formulation.
    """
    from optimization.physics_hamiltonian import PhysicalHamiltonian
    from optimization.openjij_solver import solve_sa

    all_rows = []
    topo_fn = lambda: generate_chain_topology(n_nodes=6, edge_capacity=8,
                                              memory_capacity=12)
    instances = contention_sweep_instances(topo_fn, [4, 8, 12], seed=42)

    for name, inst in instances.items():
        n_req = inst["n_requests"]
        b, ec, mc = inst["bundles"], inst["edge_capacities"], inst["memory_capacities"]

        for scale in [1.0, 10.0, 100.0, 1000.0]:
            # (a) Slack-variable QUBO via QUBOOptimizer (uses Placeholder A/B/D).
            slack_opt = __import__("optimization.qubo_optimizer",
                                   fromlist=["QUBOOptimizer"]).QUBOOptimizer(b, ec, mc)
            try:
                bqm = slack_opt.to_bqm(penalty=scale, edge_penalty=scale,
                                       memory_penalty=scale,
                                       congestion_penalty=0.0,
                                       memory_congestion_penalty=0.0)
                resp = solve_sa(bqm, num_reads=50, seed=42)
                sel = slack_opt.decode_sample(resp.first.sample, repair=True)
                slack_util = _u_sum(b, sel)
                slack_vars = bqm.num_variables
            except Exception:
                sel, slack_util, slack_vars = [], 0.0, 0

            # (b) Direct slack-free Hamiltonian (Eq. 12.1), same solver backend:
            # only |B| binary variables, no LogEncInteger slack variables.
            phys = PhysicalHamiltonian(b, ec, mc)
            try:
                q, offset = phys.to_qubo_slackfree(
                    utility_weight=1.0, one_per_request_weight=scale,
                    congestion_weight=scale, memory_ratio_weight=scale)
                dqm = dimod.BQM.from_qubo(q, offset)
                resp = solve_sa(dqm, num_reads=50, seed=42)
                var = resp.first.sample
                var["x_0"] = var.get("x_0", 0)
                sel2 = [k for k in phys.decode(var)]
                direct_util = _u_sum(b, sel2)
                direct_vars = dqm.num_variables
            except Exception:
                sel2, direct_util, direct_vars = [], 0.0, 0

            # (c) Direct Hamiltonian driven by the Metropolis annealer
            # (no slack variables by construction, incremental local moves).
            meta_sf = build_metropolis(b, ec, mc, seed=42)
            meta_r = meta_sf(penalty=scale, edge_penalty=scale, memory_penalty=scale,
                             congestion_penalty=0.0, memory_congestion_penalty=0.0,
                             max_iterations=3000, n_restarts=1, steps_per_temperature=10)
            meta_util = _u_sum(b, meta_r.get("selected", []))

            for formulation, sel, u, n_vars in [("Slack-QUBO", sel, slack_util, slack_vars),
                                                ("Direct-QUBO", sel2, direct_util, direct_vars),
                                                ("Direct-Metropolis", meta_r.get("selected", []), meta_util, len(b))]:
                # A request is served if at least one of its bundles is
                # selected; keep the highest-utility bundle per request so
                # the served ratio is comparable across formulations even
                # when the one-per-request penalty is weakly enforced.
                best = {}
                for k in sel:
                    u_k = _u_sum(b, [k])
                    if u_k > best.get(k[0], -1e18):
                        best[k[0]] = u_k
                served = len(best)
                all_rows.append({
                    "instance": name,
                    "n_requests": n_req,
                    "scale": scale,
                    "formulation": formulation,
                    "served": served,
                    "served_ratio": served / max(n_req, 1),
                    "utility": u,
                    "n_vars": n_vars,
                })

    path = os.path.join(OUT, "hamiltonian_encoding_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {path}")
    return all_rows


def _u_sum(bundles, selected):
    util_of = {(b2["request_id"], b2["bundle_id"]): b2["utility"] for b2 in bundles}
    return sum(util_of.get(k, 0.0) for k in selected)


if __name__ == "__main__":
    print("=" * 60)
    print("  Paper experiment sweep")
    print("=" * 60)

    print("\n1. Contention sweep (chain, vary cap/n_nodes)...")
    run_contention_sweep()

    print("\n2. Bond dimension sweep...")
    run_bond_dim_sweep()

    print("\n3. Grid comparison...")
    run_grid_comparison()

    print("\n4. Streaming vs batched comparison...")
    run_streaming_comparison()

    print("\n5. Hamiltonian encoding comparison (slack vs direct)...")
    run_hamiltonian_encoding_comparison()

    print(f"\nAll results in {OUT}/")



