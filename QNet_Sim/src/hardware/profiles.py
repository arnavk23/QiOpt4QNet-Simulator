"""Hardware testbed parameter profiles (paper Future Work item iv).

Bridges the idealized simulation parameters to physically realistic values for
current quantum hardware families.  Each profile captures the coherence times
(T1/T2) that govern end-to-end fidelity decay, the per-anneal duration, the
gate/readout error rates and the maximum tensor-network bond dimension the
platform can realistically sustain.

``run_hardware_profile_comparison`` replays the decoherence-aware streaming
experiment under each profile and reports how the delivered fidelity and
served utility degrade as coherence times drop.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@dataclass
class HardwareProfile:
    name: str
    t1_us: float
    t2_us: float
    anneal_time_us: float
    gate_error_rate: float
    readout_error_rate: float
    bond_dim_max: int
    description: str


IDEAL_PROFILE = HardwareProfile(
    name="ideal",
    t1_us=1e9, t2_us=1e9,
    anneal_time_us=1.0, gate_error_rate=0.0, readout_error_rate=0.0,
    bond_dim_max=512,
    description="Noiseless reference: infinite coherence, no gate/readout errors.")

SUPERCONDUCTING_PROFILE = HardwareProfile(
    name="superconducting_2024",
    t1_us=60.0, t2_us=40.0,
    anneal_time_us=200.0, gate_error_rate=1e-3, readout_error_rate=5e-2,
    bond_dim_max=128,
    description="Transmon QPUs (e.g. D-Wave/IBM-class): ~60us T1, ~40us T2, "
                "CNOT error ~1e-3, mid-circuit readout ~5e-2, anneal <=200us.")

ION_TRAP_PROFILE = HardwareProfile(
    name="ion_trap",
    t1_us=2000.0, t2_us=1000.0,
    anneal_time_us=50.0, gate_error_rate=1e-4, readout_error_rate=2e-2,
    bond_dim_max=256,
    description="Trapped-ion QCCD: coherence in the ms range, gate error ~1e-4, "
                "state prep/measurement ~2e-2.")

NEUTRAL_ATOM_PROFILE = HardwareProfile(
    name="neutral_atom",
    t1_us=200.0, t2_us=150.0,
    anneal_time_us=100.0, gate_error_rate=5e-4, readout_error_rate=5e-2,
    bond_dim_max=256,
    description="Rydberg neutral-atom arrays: ~200us coherence, native Rydberg "
                "gates ~5e-4, lower state-prep fidelity.")

PHOTONIC_PROFILE = HardwareProfile(
    name="photonic",
    t1_us=10000.0, t2_us=5000.0,
    anneal_time_us=10.0, gate_error_rate=1e-5, readout_error_rate=1e-2,
    bond_dim_max=512,
    description="Photonic/fiber-memory platforms: ms-scale storage, near-lossless "
                "gates, fast anneal; attractive long-term profile.")


ALL_PROFILES = [IDEAL_PROFILE, SUPERCONDUCTING_PROFILE, ION_TRAP_PROFILE,
                NEUTRAL_ATOM_PROFILE, PHOTONIC_PROFILE]


def run_hardware_profile_comparison(topology_fn: Callable, n_slots: int = 15,
                                    mean_rate: float = 1.5,
                                    profiles: Optional[List[HardwareProfile]] = None,
                                    seed: int = 7,
                                    out_dir: Optional[str] = None) -> List[dict]:
    """Replay the decoherence-aware experiment under each hardware profile.

    The risk-aware router is run with the profile's T2 (via ``tau_mem``) and
    T1 (via ``t1_us``); the delivered fidelity is then recomputed under the
    same coherence times, so short-coherence platforms show a clear penalty.
    """
    import csv
    from optimization.time_dependent_optimizer import run_time_dependent_comparison

    profiles = profiles or ALL_PROFILES
    rows = []
    for prof in profiles:
        res = run_time_dependent_comparison(
            topology_fn, n_slots=n_slots, mean_rate=mean_rate,
            tau_mem=prof.t2_us, t1_us=prof.t1_us, seed=seed)
        dec = res["decoherence_aware"]
        rows.append({
            "profile": prof.name,
            "t1_us": prof.t1_us,
            "t2_us": prof.t2_us,
            "anneal_time_us": prof.anneal_time_us,
            "served_ratio": dec["served_ratio"],
            "utility": dec["utility"],
            "mean_fidelity": dec["mean_fidelity"],
            "mean_delivered_fidelity": dec["mean_delivered_fidelity"],
            "fidelity_retention": dec["mean_delivered_fidelity"] / max(dec["mean_fidelity"], 1e-12),
        })
        print(f"{prof.name:>20}: served {dec['served_ratio']:.2f}, "
              f"util {dec['utility']:.1f}, delivered fid "
              f"{dec['mean_delivered_fidelity']:.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "hardware_profile_comparison.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")
    return rows


if __name__ == "__main__":
    from experiments.instances import generate_chain_topology
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                           "results", "experiments"))
    topo = lambda: generate_chain_topology(n_nodes=6, edge_capacity=6,
                                           memory_capacity=10, latency=5.0)
    run_hardware_profile_comparison(topo, n_slots=15, mean_rate=1.0,
                                    profiles=[IDEAL_PROFILE, SUPERCONDUCTING_PROFILE,
                                              ION_TRAP_PROFILE, PHOTONIC_PROFILE],
                                    out_dir=out_dir)
