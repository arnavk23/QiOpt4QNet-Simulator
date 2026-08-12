# OpenJij Penalty Calibration Benchmark

## Purpose

OpenJij SA/SQA should serve as a standardized QUBO baseline. The repository
should therefore preserve a simple conventional coefficient rule and compare it
against the proposed resource-aware rule under identical solver settings.

The calibration study should answer a narrow question:

> Does resource-aware edge/memory calibration reduce sufficient penalty
> coefficients while preserving exactness, and do those reductions affect raw
> feasibility, solution quality, or robustness under fixed solver settings?

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

The primary evidence for calibration quality is the size of the sufficient
penalty coefficients together with raw feasibility and solution quality at `1x`.
A tighter coefficient rule is useful even if heuristic solver quality is
unchanged, provided that the same hard-constraint guarantee is preserved.

## Held-out validation

A separate held-out experiment evaluates the calibration rules on instance
seeds `10` through `99`, which were not used in the initial exploratory
calibration experiments. The request-count dependence of the resource-aware
bound was predicted before the held-out evaluation: sparse reachable-load sets
were expected to permit stronger tightening, while increasing contention was
expected to make reachable loads denser and drive the minimum realizable
penalty drop toward `1`.

From `QNet_Sim`:

```bash
PYTHONPATH=src python3 src/experiments/run_penalty_calibration_heldout.py
```

A coefficient-only scan can be run with:

```bash
PYTHONPATH=src python3 src/experiments/run_penalty_calibration_heldout.py --scan-only
```

The held-out coefficient scan covers 1,080 instances across four topology
configurations, 90 fixed held-out instance seeds, and request counts `8`, `16`,
and `24`.

At the exact `1x` calibration scale:

- `B` was strictly lower under resource-aware calibration in
  `133 / 1080` instances (`12.31%`);
- `D` was strictly lower in `49 / 1080` instances (`4.54%`);
- both `B` and `D` were lower simultaneously in
  `43 / 1080` instances (`3.98%`);
- at least one of `B` or `D` was lower in
  `139 / 1080` instances (`12.87%`);
- conditional on strict tightening, `B` was reduced by `23.84%` on average;
- conditional on strict tightening, `D` was reduced by `32.81%` on average.

The 139 instances for which at least one resource penalty changed were then
evaluated using matched conventional and resource-aware QUBOs. Selection for
this solver study depended only on whether `B` or `D` changed, not on solver
performance. The remaining 941 instances had no resource-coefficient treatment
difference to evaluate.

Each selected instance used SA and SQA, five fixed solver seeds
(`101, 202, 303, 404, 505`), and 100 reads per run. CP-SAT was used as an exact
reference when optimality was certified.

The global solver results were:

- SA raw feasibility: `0.099` conventional vs. `0.086` resource-aware;
- SA repaired optimality gap: `61.01%` vs. `60.72%`;
- SQA raw feasibility: `0.285` vs. `0.281`;
- SQA raw optimality gap on feasible runs: `36.26%` vs. `36.11%`;
- SQA repaired optimality gap: `29.12%` vs. `29.82%`.

The raw optimality gap conditioned on feasibility is retained as a descriptive
metric because the conventional and resource-aware arms need not produce
feasible solutions on exactly the same runs. Raw feasible-read rate and repaired
optimality gap provide the cleaner overall comparisons.

### Paired solver analysis

The completed solver runs can be analyzed without rerunning OpenJij:

```bash
PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_heldout.py
```

Conventional and resource-aware runs are paired using the same topology,
instance seed, request count, sampler, and solver seed. The five solver-seed
differences are averaged within each instance, leaving 139 instance-level
paired comparisons per sampler.

The paired analysis found:

- SA raw-feasibility difference
  (`resource-aware - conventional`):
  mean `-0.01295`, median `0.0`,
  bootstrap 95% CI `[-0.02878, 0.00144]`;
- SA repaired-gap difference:
  mean `-0.294` percentage points, median `0.0`,
  bootstrap 95% CI `[-1.011, 0.401]`;
- SQA raw-feasibility difference:
  mean `-0.00432`, median `0.0`,
  bootstrap 95% CI `[-0.04748, 0.03885]`;
- SQA repaired-gap difference:
  mean `+0.699` percentage points, median `+0.152`,
  bootstrap 95% CI `[-0.785, 2.155]`.

All four confidence intervals include zero. The paired analysis therefore
supports the same conclusion as the pooled summaries: under the tested
fixed-budget solver settings, tightening the capacity penalties did not produce
a detectable systematic SA or SQA performance improvement.

### Reachable-load mechanism analysis

The request-count dependence of the coefficient tightening can be analyzed
without running SA, SQA, or CP-SAT:

```bash
PYTHONPATH=src python3 src/experiments/analyze_penalty_calibration_mechanism.py
```

This analysis regenerates all 1,080 held-out instances and directly measures the
minimum realizable resource-penalty drop `delta` used by the resource-aware
bound.

For edge constraints, the mean fraction of eligible bundle/resource pairs with
`delta = 1` increased with request count:

- `n = 8`: `75.32%`;
- `n = 16`: `90.93%`;
- `n = 24`: `96.95%`.

For memory constraints:

- `n = 8`: `57.75%`;
- `n = 16`: `83.71%`;
- `n = 24`: `94.13%`.

At the same time, strict coefficient tightening decreased sharply.

For `B`:

- `n = 8`: `26.11%` of instances;
- `n = 16`: `8.61%`;
- `n = 24`: `2.22%`.

For `D`:

- `n = 8`: `12.50%`;
- `n = 16`: `0.83%`;
- `n = 24`: `0.28%`.

The most direct saturation diagnostic is whether a maximum-utility bundle
(`P0`) also has a realizable `delta = 1` case. Such a pair contributes
`P0 / 1 = P0` and therefore pins the resource-aware bound to the conventional
scale.

The observed `P0`-pin frequencies were:

- edge: `73.89%`, `91.39%`, and `97.78%` for
  `n = 8, 16, 24`, respectively;
- memory: `87.50%`, `99.17%`, and `99.72%`.

Within each request-count/family group, the `P0`-pin frequency was the exact
complement of the strict-tightening frequency. The held-out data therefore
directly support the predicted saturation mechanism: increasing request count
makes unit penalty drops increasingly reachable, which progressively removes
the opportunity for resource-aware coefficient tightening.

Taken together, the held-out results support resource-aware calibration
primarily as a coefficient-tightening method. It can produce substantially
smaller sufficient edge and memory penalties when the reachable-load structure
contains useful gaps, while the advantage naturally collapses as reachable
loads become dense. The matched solver study did not detect a meaningful
overall SA/SQA performance advantage from tightening `B` and `D` alone under
the tested fixed-budget settings.

The held-out experiment and follow-up analyses write:

```text
results/penalty_calibration/heldout_coefficient_scan.csv
results/penalty_calibration/heldout_tightening_solver_runs.csv
results/penalty_calibration/heldout_tightening_solver_summary_global.csv
results/penalty_calibration/heldout_tightening_solver_summary_by_regime.csv
results/penalty_calibration/heldout_paired_analysis.csv
results/penalty_calibration/heldout_mechanism_analysis.csv
```

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

The held-out study supports resource-aware calibration as a secondary
coefficient-calibration contribution. The defensible claim is not that it
improves OpenJij solution quality, but that it gives a provably no-larger
sufficient resource-penalty calibration and can produce substantially tighter
coefficients in identifiable sparse reachable-load regimes while preserving the
same exact ground-state guarantee.

The held-out mechanism analysis directly supports the predicted explanation
for this regime dependence: as request count increases, unit penalty drops
become increasingly reachable and maximum-utility `delta = 1` cases pin the
resource-aware bound to the conventional scale. The paper should therefore
report both the frequency and magnitude of tightening and the measured
saturation mechanism.

The fixed-budget solver evaluation should be reported separately as a negative
result. Both pooled and matched instance-level analyses found no detectable
systematic SA/SQA performance advantage from tightening the edge and memory
capacity penalties while leaving the request-conflict penalty `A` on the
conventional scale.