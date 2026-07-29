# Research Findings: arXiv:2605.27425 Optimizers for QiOpt4QNet

## Setup

Compared 5 solvers on chain (4–8 nodes) and grid (3×3–4×4) topologies with 2–10 random requests per instance, across 20+ experiments.

### Solvers compared

| Solver | Type | Description |
|--------|------|-------------|
| QUBO+SA (OpenJij) | Quantum-inspired | PyQUBO-compiled QUBO solved via OpenJij simulated annealing |
| Metropolis Annealer | Quantum-inspired | Direct Metropolis–Hamiltonian MC with greedy seed, bit-flip/swap moves, geometric cooling |
| Tensor Network MPS | Quantum-inspired | Boundary-MPS compressor with SVD truncation at bond dim χ=8 |
| Utility-Density Greedy | Classical baseline | Greedy by utility/(total demand), individual-feasibility check |
| Fidelity-Aware Greedy | Classical baseline | Greedy by raw utility, individual-feasibility check |

---

## Q1: Does MPS compression scale better than SA under high contention?

**Answer: No — the current MPS implementation collapses under high contention while SA and Metropolis remain effective.**

| Contention | QUBO+SA | Metropolis | Tensor MPS |
|------------|---------|------------|------------|
| Low (2–3 req) | 2–3 served, 0 viol | 2–3 served, 0 viol | 2–3 served, 0 viol |
| Medium (4–6 req) | 2–4 served, 1–4 viol | 2–6 served, 0 viol | 0–6 served, 0 viol |
| High (8–10 req) | 3–7 served, 0–4 viol | 4–10 served, 0–1 viol | 0–9 served, 0 viol |

**Key finding:** The MPS as implemented uses only on-site utility in its local tensors; capacity-coupling penalties between requests are applied only as a Boltzmann factor during bond contraction. When many requests contend for the same edge/memory, the contraction fails to capture the necessary trade-offs, often producing all-reject solutions (0 served) in the highest-contention chain cases.

**Metropolis Annealer consistently beats QUBO+SA** in both utility and feasibility — because it directly computes the QUBO energy (including penalties) at each step rather than going through pyqubo compilation, and its greedy seed + Metropolis acceptance avoids getting stuck in infeasible minima.

---

## Q2: What bond dimension χ preserves enough correlation without exploding cost?

**Answer: χ=2–4 is sufficient for low contention; no χ value fixes the high-contention failure.**

| χ | Low contention (3 req) | Medium (5 req) | High (8 req) |
|---|----------------------|----------------|--------------|
| 1 | 88.2% of Metro utility | 100% | 0% (0 served) |
| 2 | 88.2% | 100% | 0% |
| 4 | 88.2% | 100% | 0% |
| 8 | 88.2% | 100% | 0% |
| 16 | 88.2% | 100% | 0% |

**Timing:** χ=1 → 0.06s, χ=16 → 0.78s (13× slowdown for no accuracy gain).

**Key finding:** The bond dimension does not drive accuracy in this formulation because capacity coupling is only applied as a re-weighting of the SVD rather than baked into the tensor elements themselves. The architecture needs the capacity penalties to be part of the local tensor (like a physical Hamiltonian term), not just a pairwise re-scaling during contraction. Until that redesign, χ is irrelevant for accuracy — only for runtime.

---

## Q3: Can tensor pre-screening reduce the annealer's search space?

**Answer: Yes — 62–87% bundle reduction for low-medium contention, but 0% when TN fails.**

| Instance | Bundles before | Bundles after | Reduction | Utility same? | Time saved? |
|----------|---------------|--------------|-----------|---------------|-------------|
| chain_6_req_3 | 9 | 3 | **67%** | Yes | ~Same |
| chain_6_req_5 | 13 | 5 | **62%** | Yes | ~Same |
| chain_6_req_8 | 21 | 21 | **0%** | Yes | _Slower_ |
| grid_3x4_req_3 | 23 | 3 | **87%** | Yes | _Slower_ |
| grid_3x4_req_5 | 41 | 7 | **83%** | Yes | _Slower_ |
| grid_3x4_req_8 | 59 | 59 | **0%** | Yes | _Slower_ |
| grid_4x4_req_5 | 40 | 7 | **82%** | Yes | _Slower_ |
| grid_4x4_req_10 | 54 | 10 | **81%** | ≈same | _Slower_ |

**Key finding:** Pre-screening works for low contention (high reduction, identical utility) but the pipeline is always slower than Metro-alone because TN runtime dominates. The pipeline only helps if TN runtime < reduction in Metro runtime. Currently, TN is too slow for this trade-off to pay off (TN: 0.11s vs Metro: 0.05s). For high contention, TN selects nothing so 0% reduction.

**Recommendation:** Use TN pre-screening only if TN runtime can be reduced by ~10× (e.g., via lower χ or optimized contraction), or if the problem is large enough that Metro's O(N²) conflict check dominates.

---

## Q4: How do all methods compare across topologies?

**Overall ranking by utility and feasibility:**

1. **Metropolis Annealer** — Best trade-off: highest utility, zero violations in most cases, 0.02–0.3s runtime
2. **Utility-Density Greedy** — Fastest, high utility, but often violates capacity (1–5 violations)
3. **Fidelity-Aware Greedy** — Identical to UD-Greedy in practice (same selection order for these instances)
4. **QUBO+SA (OpenJij)** — Struggles with penalties: often converges to infeasible solutions (1–5 violations) or low utility; 0.03–0.7s runtime
5. **Tensor Network MPS** — Works for low contention, fails entirely for high contention on chains; runtime grows with χ and problem size

### Runtime scaling

| Method | Small (2–3 req) | Medium (4–6 req) | Large (8–10 req) |
|--------|----------------|------------------|------------------|
| QUBO+SA | 0.03–0.25s | 0.05–0.51s | 0.06–0.72s |
| Metropolis | 0.02–0.12s | 0.03–0.17s | 0.04–0.32s |
| Tensor MPS | 0.003–0.06s | 0.01–0.48s | 0.02–1.1s |
| Greedy baselines | <0.001s | <0.001s | <0.001s |

**Metropolis Annealer is 1.5–3× faster than QUBO+SA** across all sizes, and finds better (feasible) solutions.

---

## Summary of recommendations for QiOpt4QNet

1. **Use Metropolis Annealer as the primary solver** — it consistently finds feasible, high-utility solutions faster than QUBO+SA, with direct energy computation (no pyqubo compilation overhead) and natural compatibility with the bundle-selection problem structure.

2. **MPS compressor needs a redesign** to bake capacity penalties into the local tensor Hamiltonian (not just as contraction re-weighting) before it can compete. The current formulation is only useful for low-contention scenarios where simple greedy already works.

3. **Pre-screening is viable** once TN runtime is optimized, but currently not worth the overhead. The reduction rate (62–87%) is promising — a faster TN implementation (e.g., using opt-einsum, lower χ, or skipping SVD for small bonds) could make the pipeline practical.

4. **Greedy baselines are sufficient** when speed is critical and slight capacity violations are acceptable. They serve as a strong sanity check for the more complex solvers.

5. **QUBO+SA (OpenJij)** is not recommended as the primary solver given that Metropolis Annealer matches or beats it on every metric — but it remains useful as a cross-check and for hardware-compatible QUBO output.

---

## Q5: How do the methods scale on large Waxman topologies?

**Waxman random graphs (geographic model) with 20–50 nodes, 10–30 requests.**

| Instance | Nodes | Edges | Req | QUBO+SA | Metro | TN MPS | Pipeline | UD-Greedy | FA-Greedy |
|----------|-------|-------|-----|---------|-------|--------|----------|-----------|-----------|
| waxman_20_req_10 | 20 | 38 | 10 | 4sv/4v/**120e** | **10sv/0v/-391e** | 10sv/0v/-331e | **10sv/0v/-391e** | 10sv/1v | 10sv/1v |
| waxman_30_req_15 | 30 | 24 | 15 | 3sv/3v/20e | 4sv/0v/-101e | 4sv/0v/-98e | **4sv/0v/-101e** | **7sv**/3v | **7sv**/3v |
| waxman_40_req_20 | 40 | 56 | 20 | 6sv/7v/209e | 14sv/0v/-464e | 13sv/0v/-408e | **14sv/0v/-470e** | 18sv/7v | 18sv/8v |
| waxman_50_req_30 | 50 | 42 | 30 | 5sv/4v/374e | 10sv/0v/-192e | 0sv/0v/4688e | **10sv/0v/-195e** | 18sv/16v | 18sv/16v |

*sv=served, v=violations, e=energy (lower is better). Pipeline = TN pre-screen + Metro refine.*

### Key findings

1. **QUBO+SA collapses at scale** — serves only 3–6 of up to 30 requests, with capacity violations. The QUBO search space explodes with log-encoded slack variables (up to ~150 binary variables); SA with 20 reads cannot navigate it.

2. **Metropolis Annealer is the strongest feasible solver** — serves 10–14 requests with zero violations across all instances. The adaptive temperature schedule (arXiv:2605.27425 §2) keeps the acceptance rate in the target band, avoiding both freeze-out and random-walk regimes.

3. **TN→Metro Pipeline matches Metro-alone** while running 3× faster (6s vs 19s on waxman_50). The TN pre-screen provides a near-optimal seed; the annealer then refines efficiently with fewer iterations.

4. **Tensor Network MPS alone degrades on large instances** — served 0 on waxman_50_req_30 because the sequential decoder collapsed to the all-None solution. The MPS architecture needs a proper marginal decoder (contracting the full MPS to compute per-option probabilities) rather than the current sweep-decoder.

5. **Greedy baselines serve the most requests (18) but with massive violations (7–16)** — they ignore capacity constraints entirely once contention is high.

### Runtime analysis (Waxman scale)

| Method | waxman_20 | waxman_30 | waxman_40 | waxman_50 |
|--------|-----------|-----------|-----------|-----------|
| QUBO+SA | 0.23s | 0.10s | 0.43s | 0.36s |
| Metropolis | 10.9s | 5.6s | 18.2s | 19.0s |
| Tensor MPS | 0.49s | 0.07s | 0.87s | 0.38s |
| **TN→Metro Pipeline** | **3.8s** | **1.8s** | **6.4s** | **5.9s** |
| Greedy | <0.01s | <0.01s | <0.01s | <0.01s |

**Pipeline is 2.5–3.2× faster than Metro-alone** and matches its solution quality. This makes it the most practical option for large instances where runtime matters.
