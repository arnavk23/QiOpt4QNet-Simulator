import numpy as np
import pytest

from experiments.scaling_analysis import (
    _metropolis_model, find_crossover, fit_metropolis_law, fit_mps_law,
)


def test_metropolis_model_shape():
    # quality must increase toward Q_opt with more time.
    q1 = _metropolis_model(t=1.0, q_opt=100.0, c1=2.0, c2=0.5, n=10)
    q2 = _metropolis_model(t=10.0, q_opt=100.0, c1=2.0, c2=0.5, n=10)
    assert q2 > q1
    assert q2 < 100.0


def _synthetic_metropolis_rows():
    rows = []
    for n in (10, 20):
        for t in (0.1, 0.5, 1.0, 2.0):
            q = _metropolis_model(t=t, q_opt=200.0, c1=3.0, c2=0.4, n=n)
            rows.append({"solver": "Metropolis", "n_requests": n,
                         "time_s": t, "utility": q, "bond_dim": 0})
            rows.append({"solver": "TensorNetwork", "n_requests": n,
                         "time_s": t, "utility": 200.0, "bond_dim": 32})
    return rows


def test_fit_metropolis_law_recovers_saturation():
    rows = _synthetic_metropolis_rows()
    fit = fit_metropolis_law(rows)
    assert fit["r2"] > 0.9
    assert fit["c1"] > 0 and fit["c2"] > 0


def test_fit_metropolis_flat_curve():
    rows = []
    for n in (10, 20):
        for t in (0.1, 1.0, 2.0):
            rows.append({"solver": "Metropolis", "n_requests": n,
                         "time_s": t, "utility": 150.0, "bond_dim": 0})
            rows.append({"solver": "TensorNetwork", "n_requests": n,
                         "time_s": t, "utility": 150.0, "bond_dim": 32})
    fit = fit_metropolis_law(rows)
    assert fit["note"]  # quality saturated everywhere


def test_find_crossover():
    rows = []
    for t in (0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        rows.append({"solver": "Metropolis", "n_requests": 20, "budget": t,
                     "time_s": t, "utility": 100.0 * (1 - np.exp(-t)), "bond_dim": 0})
        rows.append({"solver": "TensorNetwork", "n_requests": 20, "budget": t,
                     "time_s": t, "utility": 60.0, "bond_dim": 32})
    out = find_crossover(rows, threshold_fraction=0.9)
    assert "crossover" in out and "rows" in out  # dict, may report no crossover


def test_fit_mps_runtime_law():
    rows = []
    for n in (10, 20, 30):
        for chi in (4, 8, 16, 32):
            rows.append({"solver": "TensorNetwork", "n_requests": n,
                         "bond_dim": chi, "time_s": n * chi ** 3 * 1e-5,
                         "utility": 200.0 - 50.0 * np.exp(-0.3 * chi)})
    fit = fit_mps_law(rows)
    assert "per_n" in fit
    for n, f in fit["per_n"].items():
        assert "c5" in f and f["c5"] > 0
