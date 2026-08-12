"""Paired analysis for the held-out penalty-calibration experiment.

This script compares conventional and resource-aware calibration on matched
(instance, solver-seed) runs from the completed held-out experiment.

Run from QNet_Sim:

    PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_heldout.py
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
from collections import defaultdict


BOOTSTRAP_SEED = 20260811
BOOTSTRAP_SAMPLES = 10000
TIE_TOL = 1e-12


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values, seed):
    """Return a reproducible 95% bootstrap CI for the mean."""

    rng = random.Random(seed)
    n = len(values)

    bootstrap_means = []

    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            values[rng.randrange(n)]
            for _ in range(n)
        ]
        bootstrap_means.append(statistics.fmean(sample))

    bootstrap_means.sort()

    lower_index = int(0.025 * BOOTSTRAP_SAMPLES)
    upper_index = int(0.975 * BOOTSTRAP_SAMPLES) - 1

    return (
        bootstrap_means[lower_index],
        bootstrap_means[upper_index],
    )


def summarize_differences(
    sampler,
    metric,
    values,
    lower_is_better,
    bootstrap_seed,
):
    mean_difference = statistics.fmean(values)
    median_difference = statistics.median(values)

    ci_low, ci_high = bootstrap_mean_ci(
        values,
        bootstrap_seed,
    )

    ra_better = 0
    conventional_better = 0
    tied = 0

    for value in values:
        if abs(value) <= TIE_TOL:
            tied += 1
        elif lower_is_better:
            if value < 0:
                ra_better += 1
            else:
                conventional_better += 1
        else:
            if value > 0:
                ra_better += 1
            else:
                conventional_better += 1

    n = len(values)

    return {
        "sampler": sampler,
        "metric": metric,
        "difference_definition": "resource_aware - conventional",
        "lower_is_better": lower_is_better,
        "n_instances": n,
        "mean_paired_difference": mean_difference,
        "median_paired_difference": median_difference,
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "ra_better_fraction": ra_better / n,
        "conventional_better_fraction":
            conventional_better / n,
        "tie_fraction": tied / n,
    }


def main():
    results_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            "penalty_calibration",
        )
    )

    runs_path = os.path.join(
        results_dir,
        "heldout_tightening_solver_runs.csv",
    )

    output_path = os.path.join(
        results_dir,
        "heldout_paired_analysis.csv",
    )

    rows = read_csv(runs_path)

    matched = defaultdict(dict)

    for row in rows:
        key = (
            row["topology"],
            row["instance"],
            int(row["instance_seed"]),
            int(row["n_requests"]),
            row["sampler"],
            int(row["solver_seed"]),
        )

        matched[key][row["calibration"]] = row

    expected_pairs = len(rows) // 2

    if len(matched) != expected_pairs:
        raise RuntimeError(
            "Unexpected number of matched run keys: "
            f"{len(matched)} instead of {expected_pairs}"
        )

    instance_differences = defaultdict(
        lambda: defaultdict(list)
    )

    for key, pair in matched.items():
        if set(pair) != {
            "conventional",
            "resource_aware",
        }:
            raise RuntimeError(
                f"Incomplete calibration pair for {key}: "
                f"{sorted(pair)}"
            )

        conventional = pair["conventional"]
        resource_aware = pair["resource_aware"]

        topology, instance, instance_seed, n_requests, sampler, _ = key

        instance_key = (
            topology,
            instance,
            instance_seed,
            n_requests,
            sampler,
        )

        feasibility_difference = (
            float(resource_aware["raw_feasible_rate"])
            - float(conventional["raw_feasible_rate"])
        )

        repaired_ra = float(
            resource_aware[
                "repaired_optimality_gap_pct"
            ]
        )
        repaired_conventional = float(
            conventional[
                "repaired_optimality_gap_pct"
            ]
        )

        if (
            math.isfinite(repaired_ra)
            and math.isfinite(repaired_conventional)
        ):
            repaired_gap_difference = (
                repaired_ra - repaired_conventional
            )
        else:
            repaired_gap_difference = math.nan

        instance_differences[
            instance_key
        ]["raw_feasible_rate"].append(
            feasibility_difference
        )

        instance_differences[
            instance_key
        ]["repaired_optimality_gap_pct"].append(
            repaired_gap_difference
        )

    sampler_values = defaultdict(
        lambda: defaultdict(list)
    )

    for instance_key, metrics in instance_differences.items():
        sampler = instance_key[-1]

        feasibility_values = metrics[
            "raw_feasible_rate"
        ]

        if len(feasibility_values) != 5:
            raise RuntimeError(
                f"Expected 5 solver seeds for {instance_key}, "
                f"found {len(feasibility_values)}"
            )

        sampler_values[
            sampler
        ]["raw_feasible_rate"].append(
            statistics.fmean(feasibility_values)
        )

        repaired_values = [
            value
            for value in metrics[
                "repaired_optimality_gap_pct"
            ]
            if math.isfinite(value)
        ]

        if len(repaired_values) != 5:
            raise RuntimeError(
                "Expected 5 finite repaired gaps for "
                f"{instance_key}, found "
                f"{len(repaired_values)}"
            )

        sampler_values[
            sampler
        ]["repaired_optimality_gap_pct"].append(
            statistics.fmean(repaired_values)
        )

    summary_rows = []

    seed_offset = 0

    for sampler in ["sa", "sqa"]:
        feasibility = sampler_values[
            sampler
        ]["raw_feasible_rate"]

        repaired_gap = sampler_values[
            sampler
        ]["repaired_optimality_gap_pct"]

        summary_rows.append(
            summarize_differences(
                sampler=sampler,
                metric="raw_feasible_rate",
                values=feasibility,
                lower_is_better=False,
                bootstrap_seed=(
                    BOOTSTRAP_SEED + seed_offset
                ),
            )
        )
        seed_offset += 1

        summary_rows.append(
            summarize_differences(
                sampler=sampler,
                metric="repaired_optimality_gap_pct",
                values=repaired_gap,
                lower_is_better=True,
                bootstrap_seed=(
                    BOOTSTRAP_SEED + seed_offset
                ),
            )
        )
        seed_offset += 1

    write_csv(output_path, summary_rows)

    print("Held-out paired analysis")
    print()

    for row in summary_rows:
        print(
            f"{row['sampler'].upper():3s} "
            f"{row['metric']:30s} "
            f"n={row['n_instances']:3d} "
            f"mean-diff="
            f"{row['mean_paired_difference']:.4f} "
            f"median-diff="
            f"{row['median_paired_difference']:.4f} "
            f"95% CI=["
            f"{row['bootstrap_95_ci_low']:.4f}, "
            f"{row['bootstrap_95_ci_high']:.4f}] "
            f"RA-better="
            f"{100 * row['ra_better_fraction']:.1f}% "
            f"conv-better="
            f"{100 * row['conventional_better_fraction']:.1f}% "
            f"tie="
            f"{100 * row['tie_fraction']:.1f}%"
        )

    print()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()