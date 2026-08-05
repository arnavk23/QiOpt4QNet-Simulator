import numpy as np
import pytest

from optimization.constrained_mps import ConstrainedTensorNetworkOptimizer


def _b(bid, rid, util, edge_d, mem_d, lat=0.0, fid=0.9):
    return {
        "bundle_id": bid, "request_id": rid, "utility": util,
        "edge_demands": edge_d, "memory_demands": mem_d,
        "latency": lat, "fidelity": fid, "path": ["A", "R", "B"],
    }


def _caps():
    return {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}


def test_single_request_selects_best_bundle():
    ec, mc = _caps()
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
               _b("b1", "r1", 80.0, {("A", "R"): 2}, {"A": 2})]
    opt = ConstrainedTensorNetworkOptimizer(bundles, ec, mc)
    out = opt.solve(bond_dim=8, beta=5.0, max_sweeps=5)
    selections, util = out["selections"], out["utility"]
    assert util == pytest.approx(80.0, abs=1e-6)
    assert selections.get("r1") == "b1"


def test_capacity_constraints_respected():
    ec, mc = _caps()
    bundles = [_b("b0", "r1", 50.0, {("A", "R"): 10}, {"A": 10}),
               _b("b1", "r1", 80.0, {("A", "R"): 12}, {"A": 12})]
    opt = ConstrainedTensorNetworkOptimizer(bundles, ec, mc)
    out = opt.solve(bond_dim=8, beta=5.0, max_sweeps=5)
    selections, util = out["selections"], out["utility"]
    assert util == pytest.approx(50.0, abs=1e-6)
    assert selections.get("r1") == "b0"


def test_pairwise_capacity_conflict():
    # Two requests that cannot both fit on the shared edge: at most one is served.
    ec, mc = _caps()
    bundles = [_b("b0", "r1", 60.0, {("A", "R"): 7}, {"A": 7}),
               _b("b0", "r2", 70.0, {("A", "R"): 7}, {"A": 7})]
    opt = ConstrainedTensorNetworkOptimizer(bundles, ec, mc)
    out = opt.solve(bond_dim=8, beta=5.0, max_sweeps=5)
    selections, util = out["selections"], out["utility"]
    served = [rid for rid, bid in selections.items() if bid is not None]
    assert len(served) == 1
    assert util in (60.0, 70.0)


def test_feasibility_by_construction():
    ec, mc = _caps()
    bundles = [_b("b0", "r1", 60.0, {("A", "R"): 7}, {"A": 7}),
               _b("b1", "r1", 90.0, {("A", "R"): 12}, {"A": 12}),
               _b("b0", "r2", 70.0, {("A", "R"): 7}, {"A": 7})]
    opt = ConstrainedTensorNetworkOptimizer(bundles, ec, mc)
    out = opt.solve(bond_dim=8, beta=5.0, max_sweeps=5)
    selections = out["selections"]
    edge_load = sum(d for rid, bid in selections.items()
                    for d in opt._edge_of.get((rid, bid), {}).values())
    assert edge_load <= 10
