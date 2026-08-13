from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RESULTS_DIR = Path("results/penalty_calibration")
FIGURES_DIR = RESULTS_DIR / "figures"

MECHANISM_CSV = RESULTS_DIR / "heldout_mechanism_analysis.csv"
COEFFICIENT_SCAN_CSV = RESULTS_DIR / "heldout_coefficient_scan.csv"
PAIRED_ANALYSIS_CSV = RESULTS_DIR / "heldout_paired_analysis.csv"

FAMILY_COLORS = {
    "B": "#4477AA",
    "D": "#EE8833",
}
SAMPLER_COLORS = {
    "sa": "#4477AA",
    "sqa": "#EE8833",
}

FIG_WIDTH = 7.2
GRID_COLOR = "#E6E6E6"
SPINE_COLOR = "#333333"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require(mapping: dict, key, description: str):
    """Fetch a grouped row, failing with a message that names what is missing."""
    if key not in mapping:
        raise KeyError(
            f"Missing {description} for {key!r}. "
            f"Available keys: {sorted(mapping)}"
        )
    return mapping[key]


def apply_common_axis_style(ax, *, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.45)
    ax.tick_params(
        axis="both",
        color=SPINE_COLOR,
        width=0.7,
        length=2.8,
        labelsize=7.5,
    )
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.7)


def apply_legend_style(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D8D8D8")
    frame.set_linewidth(0.6)
    frame.set_alpha(0.95)


def save_figure(fig, stem: str, *, tight_rect=None) -> None:
    fig.tight_layout(pad=0.45, w_pad=1.0, rect=tight_rect)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_figure_1(mechanism_rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[int, str], dict[str, str]] = {}
    for row in mechanism_rows:
        n_requests = int(row["n_requests"])
        family = row["family"].strip().lower()
        grouped[(n_requests, family)] = row

    request_counts = sorted({int(row["n_requests"]) for row in mechanism_rows})

    def series(family: str, column: str) -> list[float]:
        return [
            100.0 * float(require(grouped, (n, family), "mechanism row")[column])
            for n in request_counts
        ]

    edge_tightening = series("edge", "coefficient_tightened_fraction")
    memory_tightening = series("memory", "coefficient_tightened_fraction")
    edge_delta1 = series("edge", "mean_instance_delta1_fraction")
    memory_delta1 = series("memory", "mean_instance_delta1_fraction")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.65))

    ax = axes[0]
    ax.plot(
        request_counts,
        edge_tightening,
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        color=FAMILY_COLORS["B"],
        label="Edge penalty $B$",
    )
    ax.plot(
        request_counts,
        memory_tightening,
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        color=FAMILY_COLORS["D"],
        label="Memory penalty $D$",
    )
    ax.set_xlabel("Number of requests")
    ax.set_ylabel("Instances tightened (%)")
    ax.set_title("Tightening frequency")
    ax.set_xticks(request_counts)
    ax.set_ylim(0, max(edge_tightening + memory_tightening) * 1.15)
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )
    apply_legend_style(legend)
    apply_common_axis_style(ax)

    ax = axes[1]
    ax.plot(
        request_counts,
        edge_delta1,
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        color=FAMILY_COLORS["B"],
    )
    ax.plot(
        request_counts,
        memory_delta1,
        marker="s",
        markersize=3.5,
        linewidth=1.0,
        color=FAMILY_COLORS["D"],
    )
    ax.set_xlabel("Number of requests")
    ax.set_ylabel("Violating resource–bundle pairs (%)")
    ax.set_title(r"Minimum penalty-drop frequency ($\delta=1$)")
    ax.set_xticks(request_counts)
    ax.set_ylim(0, 100)
    apply_common_axis_style(ax)

    save_figure(fig, "figure1_mechanism_summary")
    plt.close(fig)


def make_figure_2(scan_rows: list[dict[str, str]]) -> None:
    b_all = sorted(float(row["B_ratio"]) for row in scan_rows)
    d_all = sorted(float(row["D_ratio"]) for row in scan_rows)

    b_reductions = sorted(
        100.0 * (1.0 - float(row["B_ratio"]))
        for row in scan_rows
        if as_bool(row["B_tighter"])
    )
    d_reductions = sorted(
        100.0 * (1.0 - float(row["D_ratio"]))
        for row in scan_rows
        if as_bool(row["D_tighter"])
    )

    b_all_ranks = list(range(1, len(b_all) + 1))
    d_all_ranks = list(range(1, len(d_all) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.7))

    ax = axes[0]
    ax.plot(
        b_all_ranks,
        b_all,
        color=FAMILY_COLORS["B"],
        linewidth=1.0,
        label="Edge penalty $B$",
    )
    ax.plot(
        d_all_ranks,
        d_all,
        color=FAMILY_COLORS["D"],
        linewidth=1.0,
        label="Memory penalty $D$",
    )
    ax.axhline(
        1.0,
        color="0.2",
        linestyle="--",
        linewidth=0.8,
        zorder=4,
    )
    max_rank = max(len(b_all), len(d_all))
    rank_ticks = sorted(
        {
            1,
            round(0.25 * max_rank),
            round(0.50 * max_rank),
            round(0.75 * max_rank),
            max_rank,
        }
    )
    ax.set_xlabel("Held-out cases, ordered by ratio")
    ax.set_ylabel("Coefficient relative to utility-scale baseline")
    ax.set_title("Coefficient ratios")
    ax.set_xlim(1, max_rank)
    ax.set_xticks(rank_ticks)
    ax.set_ylim(0, 1.03)
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )
    apply_legend_style(legend)
    apply_common_axis_style(ax)

    ax = axes[1]
    boxplot = ax.boxplot(
        [b_reductions, d_reductions],
        positions=[1.0, 2.0],
        widths=0.40,
        patch_artist=True,
        whis=1.5,
        showcaps=False,
        showfliers=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 3.2,
        },
        medianprops={"color": "0.15", "linewidth": 0.8},
        whiskerprops={"color": "0.35", "linewidth": 0.8},
        capprops={"color": "0.35", "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markeredgewidth": 0.7,
            "markersize": 2.8,
        },
    )
    for box, color in zip(boxplot["boxes"], [FAMILY_COLORS["B"], FAMILY_COLORS["D"]]):
        box.set_facecolor(color)
        box.set_edgecolor("0.35")
        box.set_alpha(0.66)
        box.set_linewidth(0.8)

    ax.set_ylabel("Reduction from utility-scale baseline (%)")
    ax.set_title("Reduction when tightened")
    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(
        [
            "Edge penalty $B$",
            "Memory penalty $D$",
        ],
    )
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(-5, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    apply_common_axis_style(ax, grid_axis="y")

    save_figure(fig, "figure2_coefficient_ratios")
    plt.close(fig)


def make_figure_3(paired_rows: list[dict[str, str]]) -> None:
    grouped = {
        (row["sampler"].strip().lower(), row["metric"]): row for row in paired_rows
    }

    metrics = [
        {
            "key": "raw_feasible_rate",
            "title": "Raw feasibility",
            "xlabel": "Change vs. utility-scale (pp)",
            "multiplier": 100.0,
        },
        {
            "key": "raw_mean_overload_units",
            "title": "Raw overload",
            "xlabel": "Reduction vs. utility-scale (resource units/sample)",
            "multiplier": -1.0,
        },
        {
            "key": "repaired_optimality_gap_pct",
            "title": "Repaired optimality gap",
            "xlabel": "Reduction vs. utility-scale (pp)",
            "multiplier": -1.0,
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.45))
    y_positions = {"sa": 1.0, "sqa": 0.0}

    for ax, metric in zip(axes, metrics):
        plotted_values = []

        for sampler in ["sa", "sqa"]:
            row = require(grouped, (sampler, metric["key"]), "paired-analysis row")
            multiplier = metric["multiplier"]
            estimate = multiplier * float(row["mean_paired_difference"])
            ci_low, ci_high = sorted(
                [
                    multiplier * float(row["bootstrap_95_ci_low"]),
                    multiplier * float(row["bootstrap_95_ci_high"]),
                ]
            )
            plotted_values.extend([ci_low, estimate, ci_high])

            ax.errorbar(
                estimate,
                y_positions[sampler],
                xerr=[[estimate - ci_low], [ci_high - estimate]],
                fmt="o",
                color=SAMPLER_COLORS[sampler],
                markersize=3.8,
                capsize=2.2,
                capthick=0.8,
                elinewidth=0.9,
            )

        limit = max(abs(value) for value in plotted_values)
        padding = max(0.18 * limit, 0.03)
        ax.set_xlim(-(limit + padding), limit + padding)
        ax.axvline(0.0, color="0.25", linestyle="--", linewidth=0.8)
        ax.set_title(metric["title"])
        ax.set_xlabel(metric["xlabel"])
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["SQA", "SA"])
        ax.set_ylim(-0.55, 1.55)
        apply_common_axis_style(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SAMPLER_COLORS[sampler],
            marker="o",
            markersize=3.8,
            linewidth=0.9,
            label=sampler.upper(),
        )
        for sampler in ["sa", "sqa"]
    ]
    legend = axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        handlelength=1.7,
        borderpad=0.35,
        labelspacing=0.35,
    )
    apply_legend_style(legend)

    save_figure(fig, "figure3_solver_paired_effects")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    make_figure_1(load_csv_rows(MECHANISM_CSV))
    make_figure_2(load_csv_rows(COEFFICIENT_SCAN_CSV))
    make_figure_3(load_csv_rows(PAIRED_ANALYSIS_CSV))


if __name__ == "__main__":
    main()
