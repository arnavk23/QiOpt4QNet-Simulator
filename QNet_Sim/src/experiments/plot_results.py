import csv, os, math
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "experiments"))
OUT = os.path.join(DATA, "figures")
os.makedirs(OUT, exist_ok=True)


def _load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _agg(rows, group_keys, val_key, agg_fn="mean"):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys) if isinstance(group_keys, (list, tuple)) else r[group_keys]
        groups[key].append(float(r[val_key]))
    result = {}
    for k, vals in groups.items():
        if agg_fn == "mean":
            result[k] = sum(vals) / len(vals)
        elif agg_fn == "sum":
            result[k] = sum(vals)
        elif agg_fn == "std":
            result[k] = np.std(vals) if len(vals) > 1 else 0.0
    return result


def fig_contention_scaling():
    if not HAS_MPL:
        print("matplotlib not available")
        return
    rows = _load(os.path.join(DATA, "contention_sweep.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = {"Metropolis": "#1f77b4", "TensorNetwork": "#ff7f0e"}
    markers = {"Metropolis": "o", "TensorNetwork": "s"}
    styles = {"Metropolis": "-", "TensorNetwork": "--"}

    for col, cap in enumerate([4, 6, 10]):
        for row, metric in enumerate(["served_ratio", "time_s"]):
            ax = axes[row][col]
            for solver in ["Metropolis", "TensorNetwork"]:
                subset = [r for r in rows
                          if int(r["edge_capacity"]) == cap
                          and r["solver"] == solver]
                by_n = _agg(subset, ["n_requests", "n_nodes"], metric)
                for n_nodes in [4, 6, 10]:
                    pts = [(int(k[0]), v) for k, v in by_n.items() if int(k[1]) == n_nodes]
                    if not pts:
                        continue
                    pts.sort(key=lambda x: x[0])
                    xs, ys = zip(*pts)
                    label = f"{solver} N={n_nodes}" if col == 0 else None
                    ax.plot(xs, ys, label=label,
                            color=colors[solver], marker=markers[solver],
                            linestyle=styles[solver], markersize=4)

            ax.set_xlabel("Number of requests")
            if metric == "served_ratio":
                ax.set_ylabel("Served ratio")
                ax.set_ylim(0, 1.05)
                ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
            else:
                ax.set_ylabel("Wall-clock time (s)")
                ax.set_yscale("log")
            ax.set_title(f"Edge capacity = {cap}")
            ax.legend(fontsize=7) if col == 0 else None
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "contention_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_bond_dimension_scaling():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "bond_dim_sweep.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
    bond_dims = sorted(set(int(r["bond_dim"]) for r in rows if int(r["bond_dim"]) > 0))

    n_requests = sorted(set(int(r["n_requests"]) for r in rows))
    for i, bd in enumerate(bond_dims):
        subset = [r for r in rows if int(r["bond_dim"]) == bd]
        by_n = _agg(subset, "n_requests", "served_ratio")
        pts = sorted((int(k), v) for k, v in by_n.items())
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker="o", color=colors[i], label=f"$\\chi={bd}$")

    meta = [r for r in rows if r["solver"] == "Metropolis-SA"]
    by_n = _agg(meta, "n_requests", "served_ratio")
    pts = sorted((int(k), v) for k, v in by_n.items())
    xs, ys = zip(*pts)
    ax1.plot(xs, ys, marker="s", color="red", linewidth=2, label="Metropolis", linestyle="--")

    ax1.set_xlabel("Number of requests")
    ax1.set_ylabel("Served ratio")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Solution quality vs bond dimension")

    for i, bd in enumerate(bond_dims):
        subset = [r for r in rows if int(r["bond_dim"]) == bd]
        by_n = _agg(subset, "n_requests", "time_s")
        pts = sorted((int(k), v) for k, v in by_n.items())
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, marker="o", color=colors[i], label=f"$\\chi={bd}$")

    meta_by_n = _agg(meta, "n_requests", "time_s")
    pts = sorted((int(k), v) for k, v in meta_by_n.items())
    xs, ys = zip(*pts)
    ax2.plot(xs, ys, marker="s", color="red", linewidth=2, label="Metropolis", linestyle="--")

    ax2.set_xlabel("Number of requests")
    ax2.set_ylabel("Wall-clock time (s)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Runtime vs bond dimension")

    plt.tight_layout()
    path = os.path.join(OUT, "bond_dimension_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_topology_comparison():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "grid_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    topologies = sorted(set(r["topology"] for r in rows))
    x = range(len(topologies))
    width = 0.35

    for idx, metric in enumerate(["served_ratio", "time_s"]):
        ax = [ax1, ax2][idx]
        for si, solver in enumerate(["Metropolis", "TensorNetwork"]):
            vals = []
            for topo in topologies:
                subset = [r for r in rows if r["topology"] == topo and r["solver"] == solver]
                if metric == "served_ratio":
                    v = sum(float(r["served"]) for r in subset) / max(sum(int(r["n_requests"]) for r in subset), 1)
                else:
                    v = sum(float(r["time_s"]) for r in subset) / max(len(subset), 1)
                vals.append(v)
            offset = (si - 0.5) * width
            bars = ax.bar([xi + offset for xi in x], vals, width,
                          label=solver, color=["#1f77b4", "#ff7f0e"][si], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(topologies)
        if metric == "served_ratio":
            ax.set_ylabel("Served ratio")
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylabel("Mean wall-clock time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "topology_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_streaming_throughput():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "streaming_comparison.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    solvers = ["Streaming", "Batched"]
    colors = {"Streaming": "#2ca02c", "Batched": "#1f77b4"}
    markers = {"Streaming": "^", "Batched": "o"}

    for metric, ax in [("served_ratio", ax1), ("time_s", ax2)]:
        for solver in solvers:
            subset = [r for r in rows if r["solver"] == solver]
            pts = sorted((int(r["n_requests"]), float(r[metric])) for r in subset)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=markers[solver], color=colors[solver],
                    label=solver, linewidth=2, markersize=6)

        ax.set_xlabel("Number of requests")
        if metric == "served_ratio":
            ax.set_ylabel("Served ratio")
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylabel("Wall-clock time (s)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "streaming_throughput.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def generate_tables():
    rows = _load(os.path.join(DATA, "contention_sweep.csv"))
    rows_bd = _load(os.path.join(DATA, "bond_dim_sweep.csv"))
    rows_grid = _load(os.path.join(DATA, "grid_comparison.csv"))
    rows_stream = _load(os.path.join(DATA, "streaming_comparison.csv"))

    lines = []

    lines.append("=== Table 1: Contention scaling (chain, cap=6, N=10) ===")
    hdr = f"{'Req':>5} {'Metro served':>14} {'TN served':>12} {'Metro time':>12} {'TN time':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    subset = [r for r in rows if int(r["edge_capacity"]) == 6 and int(r["n_nodes"]) == 10]
    by_n = defaultdict(lambda: {"Metropolis": [], "TensorNetwork": []})
    for r in subset:
        by_n[int(r["n_requests"])][r["solver"]].append(r)
    for n in sorted(by_n.keys()):
        m = by_n[n]["Metropolis"]
        t = by_n[n]["TensorNetwork"]
        if m and t:
            ms = sum(float(r["served_ratio"]) for r in m) / len(m)
            ts = sum(float(r["served_ratio"]) for r in t) / len(t)
            mt = sum(float(r["time_s"]) for r in m) / len(m)
            tt = sum(float(r["time_s"]) for r in t) / len(t)
            lines.append(f"{n:>5} {ms:>14.3f} {ts:>12.3f} {mt:>12.4f} {tt:>12.4f}")

    lines.append("")
    lines.append("=== Table 2: Bond dimension analysis (chain_8) ===")
    hdr = f"{'Req':>5} {'χ=2 served':>14} {'χ=8 served':>14} {'χ=32 served':>15} {'Metro served':>15}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    by_n_bd = defaultdict(lambda: defaultdict(list))
    for r in rows_bd:
        by_n_bd[int(r["n_requests"])][r["solver"]].append(r)
    for n in sorted(by_n_bd.keys()):
        d = by_n_bd[n]
        def _sv(s):
            v = d.get(s, [])
            return sum(float(x["served_ratio"]) for x in v) / max(len(v), 1)
        lines.append(f"{n:>5} {_sv('TN(χ=2)'):>14.3f} {_sv('TN(χ=8)'):>14.3f} "
                     f"{_sv('TN(χ=32)'):>15.3f} {_sv('Metropolis-SA'):>15.3f}")

    lines.append("")
    lines.append("=== Table 3: Streaming vs batched ===")
    hdr = f"{'Req':>5} {'Stream served':>15} {'Batch served':>14} {'Stream time':>13} {'Batch time':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    by_n_s = defaultdict(lambda: defaultdict(list))
    for r in rows_stream:
        by_n_s[int(r["n_requests"])][r["solver"]].append(r)
    for n in sorted(by_n_s.keys()):
        d = by_n_s[n]
        ss = sum(float(x["served_ratio"]) for x in d.get("Streaming", [])) / max(len(d.get("Streaming", [])), 1)
        bs = sum(float(x["served_ratio"]) for x in d.get("Batched", [])) / max(len(d.get("Batched", [])), 1)
        st = sum(float(x["time_s"]) for x in d.get("Streaming", [])) / max(len(d.get("Streaming", [])), 1)
        bt = sum(float(x["time_s"]) for x in d.get("Batched", [])) / max(len(d.get("Batched", [])), 1)
        lines.append(f"{n:>5} {ss:>15.3f} {bs:>14.3f} {st:>13.4f} {bt:>12.4f}")

    path = os.path.join(OUT, "summary_tables.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")
    print("\n".join(lines))


def fig_hamiltonian_encoding():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "hamiltonian_encoding_comparison.csv"))
    forms = ["Slack-QUBO", "Direct-QUBO", "Direct-Metropolis"]
    colors = {"Slack-QUBO": "#d62728", "Direct-QUBO": "#2ca02c", "Direct-Metropolis": "#1f77b4"}
    instances = sorted(set(r["instance"] for r in rows), key=lambda s: int(s[1:]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    x = range(len(instances))
    width = 0.26
    for fi, form in enumerate(forms):
        nvars = []
        for inst in instances:
            sub = [r for r in rows if r["formulation"] == form and r["instance"] == inst
                   and r["scale"] == "1.0"]
            nvars.append(float(sub[0]["n_vars"]) if sub else 0.0)
        ax1.bar([xi + (fi - 1) * width for xi in x], nvars, width,
                label=form, color=colors[form], alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(instances)
    ax1.set_xlabel("Instance (requests)")
    ax1.set_ylabel("QUBO variables")
    ax1.set_title("Encoding size (scale = 1)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    scales = sorted(set(float(r["scale"]) for r in rows))
    for fi, form in enumerate(forms):
        for inst in instances:
            sub = sorted([r for r in rows if r["formulation"] == form and r["instance"] == inst],
                         key=lambda r: float(r["scale"]))
            xs = [float(r["scale"]) for r in sub]
            ys = [float(r["served_ratio"]) for r in sub]
            ax2.plot(xs, ys, marker="o", color=colors[form], markersize=4,
                     linestyle="-", alpha=0.85 if fi == 1 else 0.45,
                     label=form if inst == instances[0] else None)
    ax2.set_xscale("log")
    ax2.set_xlabel("Demand scale")
    ax2.set_ylabel("Served ratio")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Scale robustness of each encoding")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(OUT, "hamiltonian_encoding.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_scaling_analysis():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "scaling_curves.csv"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for n in sorted(set(int(r["n_requests"]) for r in rows if r["solver"] == "Metropolis")):
        sub = [r for r in rows if r["solver"] == "Metropolis" and int(r["n_requests"]) == n]
        pts = sorted((int(r["budget"]), float(r["utility"])) for r in sub)
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker="o", label=f"N={n}")

    for n in sorted(set(int(r["n_requests"]) for r in rows if r["solver"] == "TensorNetwork")):
        sub = [r for r in rows if r["solver"] == "TensorNetwork" and int(r["n_requests"]) == n]
        pts = sorted((int(r["bond_dim"]), float(r["utility"])) for r in sub)
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, marker="s", linestyle="--", label=f"N={n}")

    ax1.set_xscale("log")
    ax1.set_xlabel("Metropolis budget (steps)")
    ax1.set_ylabel("Aggregate utility")
    ax1.set_title("Metropolis quality vs budget")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale("log")
    ax2.set_xlabel("MPS bond dimension $\\chi$")
    ax2.set_ylabel("Aggregate utility")
    ax2.set_title("Tensor-network quality vs $\\chi$")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "scaling_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_time_dependent():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "time_dependent_slots.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    slots = [int(r["slot"]) for r in rows]
    dec = [int(r["served_dec"]) for r in rows]
    st = [int(r["served_static"]) for r in rows]

    width = 0.4
    ax1.bar([s - width / 2 for s in slots], dec, width, label="Decoherence-aware",
            color="#1f77b4", alpha=0.85)
    ax1.bar([s + width / 2 for s in slots], st, width, label="Static",
            color="#ff7f0e", alpha=0.85)
    ax1.set_xlabel("Time slot")
    ax1.set_ylabel("Requests served")
    ax1.set_title("Per-slot routing decisions")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    cdec = []
    cst = []
    acc = 0
    for v in dec:
        acc += v
        cdec.append(acc)
    acc = 0
    for v in st:
        acc += v
        cst.append(acc)
    ax2.plot(slots, cdec, marker="o", color="#1f77b4", label="Decoherence-aware")
    ax2.plot(slots, cst, marker="s", color="#ff7f0e", label="Static")
    ax2.set_xlabel("Time slot")
    ax2.set_ylabel("Cumulative requests served")
    ax2.set_title("Cumulative throughput")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "time_dependent.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_risk_tradeoff():
    if not HAS_MPL:
        return
    path_csv = os.path.join(DATA, "risk_gain_tradeoff.csv")
    if not os.path.exists(path_csv):
        print("risk_gain_tradeoff.csv missing; skipping")
        return
    rows = _load(path_csv)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for key, label, color in [("dec", "Decoherence-aware", "#1f77b4"),
                              ("static", "Static", "#ff7f0e")]:
        xs = [float(r[f"fid_{key}"]) for r in rows]
        ys = [float(r[f"utility_{key}"]) for r in rows]
        ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel("Mean delivered fidelity")
    ax.set_ylabel("Aggregate utility")
    ax.set_title("Fidelity-utility tradeoff under risk gain sweep")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "risk_tradeoff.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_constrained_mps():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "constrained_mps_comparison.csv"))
    ns = sorted(set(int(r["n_requests"]) for r in rows))
    solvers = ["Penalty-MPS", "Constrained-MPS"]
    colors = {"Penalty-MPS": "#1f77b4", "Constrained-MPS": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    x = range(len(ns))
    width = 0.36
    for idx, (metric, ylab, title) in enumerate([
        ("served_ratio", "Served ratio", "Solution quality"),
        ("time_s", "Wall-clock time (s)", "Runtime"),
        ("utility", "Aggregate utility", "Aggregate utility"),
    ]):
        ax = axes[idx]
        for si, solver in enumerate(solvers):
            vals = []
            for n in ns:
                sub = [r for r in rows if r["solver"] == solver and int(r["n_requests"]) == n]
                vals.append(sum(float(r[metric]) for r in sub) / max(len(sub), 1))
            ax.bar([xi + (si - 0.5) * width for xi in x], vals, width,
                   label=solver, color=colors[solver], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"N={n}" for n in ns])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "constrained_mps.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_qlearning():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "qlearning_comparison.csv"))
    eps = [int(r["episode"]) for r in rows]
    served = [int(r["served"]) for r in rows]
    reward = [float(r["reward"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(eps, served, marker="o", color="#1f77b4", label="Served requests")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Served requests")
    ax1.set_title("Q-learning router training progress")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eps, reward, marker="s", color="#d62728", alpha=0.7, label="Episode reward")
    ax2.set_ylabel("Episode reward", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    plt.tight_layout()
    path = os.path.join(OUT, "qlearning.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def fig_hardware_profiles():
    if not HAS_MPL:
        return
    rows = _load(os.path.join(DATA, "hardware_profile_comparison.csv"))
    profiles = [r["profile"] for r in rows]
    fid = [float(r["mean_delivered_fidelity"]) for r in rows]
    ret = [float(r["fidelity_retention"]) for r in rows]
    raw = [float(r["mean_fidelity"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    x = range(len(profiles))
    width = 0.38
    ax1.bar([xi - width / 2 for xi in x], raw, width, label="Generated fidelity",
            color="#1f77b4", alpha=0.7)
    ax1.bar([xi + width / 2 for xi in x], fid, width, label="Delivered fidelity",
            color="#2ca02c", alpha=0.85)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(profiles, rotation=12, fontsize=8)
    ax1.set_ylabel("Mean fidelity")
    ax1.set_title("Fidelity loss across hardware profiles")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, [r * 100 for r in ret], color="#9467bd", alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(profiles, rotation=12, fontsize=8)
    ax2.set_ylabel("Fidelity retention (%)")
    ax2.set_title("Retention = delivered / generated")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUT, "hardware_profiles.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("Generating figures...")
    fig_contention_scaling()
    fig_bond_dimension_scaling()
    fig_topology_comparison()
    fig_streaming_throughput()
    fig_hamiltonian_encoding()
    fig_scaling_analysis()
    fig_time_dependent()
    fig_risk_tradeoff()
    fig_constrained_mps()
    fig_qlearning()
    fig_hardware_profiles()
    print("\nGenerating tables...")
    generate_tables()
    print(f"\nAll outputs in {OUT}/")
