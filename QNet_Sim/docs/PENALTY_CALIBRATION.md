# OpenJij Penalty Calibration Benchmark

## Purpose

OpenJij SA/SQA should serve as a standardized QUBO baseline. The repository
should therefore preserve a simple conventional coefficient rule and compare it
against the proposed resource-aware rule under identical solver settings.

The calibration study should answer a narrow question:

> Does resource-aware edge/memory calibration improve raw feasibility,
> solution quality, robustness, or coefficient scale without relying on
> per-instance manual tuning?

## Calibration variants

### Conventional reference

`conventional_coefficients()` sets the hard request, edge, and memory penalties
just above the largest positive bundle utility. This is intentionally simple
and transparent.

### Resource-aware proposal

`proposed_global_coefficients()` leaves the request-conflict penalty `A` on the
same conventional scale, but calculates separate `B` and `D` values from the
edge/memory capacities, bundle demands, reachable competing loads, and bundle
utilities.

The implementation uses bounded dynamic programming to avoid the exponential
full-load enumeration used in the initial version.

### Soft congestion is separate

`C` and `E` are soft load-balancing/congestion regularizers. They are **not**
part of the hard-constraint calibration claim. For the clean calibration
ablation, run with `C = E = 0`. Congestion can then be restored in a separate
experiment.

## Main experiment

From `QNet_Sim`:

```bash
PYTHONPATH=src python src/experiments/run_penalty_calibration.py
```

This runs the quick comparison over chain/grid instances using:

- OpenJij SA and SQA;
- conventional and resource-aware calibration;
- coefficient sensitivity multipliers `0.25x, 0.5x, 1x, 2x, 4x`;
- multiple independently generated network/request instances;
- CP-SAT as the exact reference when OR-Tools is installed.

A larger paper-style sweep is:

```bash
PYTHONPATH=src python src/experiments/run_penalty_calibration.py --full --reads 100
```

Outputs are written to:

```text
results/penalty_calibration/calibration_runs.csv
results/penalty_calibration/calibration_summary.csv
```

## Metrics that matter

Do not judge calibration only after feasibility repair. Repair can hide poor
penalty choices.

The benchmark therefore reports:

- raw feasible-read rate;
- raw request conflicts;
- raw edge/memory violations and overload;
- best raw-feasible utility;
- repaired utility as a secondary operational metric;
- CP-SAT optimality gap where available;
- coefficient magnitudes;
- QUBO compile/calibration time;
- SA/SQA sampling time.

The primary evidence for a better calibrator is improved **raw** feasibility and
quality at `1x`, plus a wider useful range in the sensitivity sweep.

## Tests

```bash
PYTHONPATH=src pytest tests/test_penalty_calibration.py -v
```

The tests cover utility-scale calibration, reachable-load calculation,
separate edge/memory bounds, no-overload cases, sensitivity scaling, and API
validation.

## Recommended paper framing

Use:

- **Conventional calibration** = required reference baseline.
- **Resource-aware calibration** = proposed method/ablation.
- **OpenJij SA/SQA** = standardized QUBO solver family, not itself the novel
  contribution.

Only elevate resource-aware calibration to a paper contribution if it shows a
consistent advantage over the conventional rule across independent instances,
loads, topologies, and solver seeds.