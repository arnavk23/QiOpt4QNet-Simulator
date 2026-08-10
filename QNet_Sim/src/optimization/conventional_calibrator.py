"""Conventional penalty calibration for the QiOpt4QNet QUBO.

This module intentionally provides a simple, transparent reference rule:
set every hard-constraint penalty just above the largest positive bundle
utility. It is the baseline against which resource-aware calibration should
be compared; it is not presented as a novel method.
"""

from __future__ import annotations

from typing import Dict


def penalty_epsilon(scale: float) -> float:
    """Return a small strictly-positive margin on the supplied utility scale."""
    scale = max(0.0, float(scale))
    return max(1e-9, 1e-6 * scale)


def positive_utility_scale(optimizer) -> float:
    """Largest positive bundle utility, or zero when no bundle is profitable."""
    return max(
        (max(0.0, float(bundle["utility"])) for bundle in optimizer.bundles),
        default=0.0,
    )


def conventional_coefficients(
    optimizer,
    *,
    safety_factor: float = 1.0,
    congestion_penalty: float = 0.0,
    memory_congestion_penalty: float = 0.0,
) -> Dict[str, float]:
    """Calibrate A/B/D from the global utility scale.

    A controls the at-most-one-bundle-per-request conflict penalty.
    B and D control edge- and memory-capacity penalties.
    C and E are soft congestion regularizers and are deliberately kept
    separate from hard-constraint calibration.
    """
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")

    p0 = positive_utility_scale(optimizer)
    hard = safety_factor * p0 + penalty_epsilon(p0)

    return {
        "A": hard,
        "B": hard,
        "C": float(congestion_penalty),
        "D": hard,
        "E": float(memory_congestion_penalty),
    }