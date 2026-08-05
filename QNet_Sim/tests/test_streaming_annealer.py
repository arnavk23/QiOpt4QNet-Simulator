from optimization.streaming_annealer import StreamingAnnealer


def _b(bid, rid, util, edge_d, mem_d, lat=0.0):
    return {
        "bundle_id": bid, "request_id": rid, "utility": util,
        "edge_demands": edge_d, "memory_demands": mem_d,
        "latency": lat, "path": ["A", "R", "B"],
    }


def _caps():
    return {("A", "R"): 10, ("R", "B"): 10}, {"A": 10, "R": 10, "B": 10}


def test_empty():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42)
    assert sa.active_count() == 0
    assert sa.get_selected() == []


def test_add_single_request():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42)
    bundles = [
        _b("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _b("b1", "r1", 10.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
    ]
    sa.add_request("r1", bundles)
    assert sa.active_count() == 1
    sel = sa.get_selected()
    assert len(sel) == 1
    assert sel[0][1] == "b0"


def test_add_two_requests():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42)
    sa.add_request("r1", [
        _b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
    ])
    sa.add_request("r2", [
        _b("b0", "r2", 40.0, {("A", "R"): 1}, {"A": 1}),
    ])
    assert sa.active_count() == 2
    assert len(sa.get_selected()) == 2


def test_remove_request():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42)
    sa.add_request("r1", [
        _b("b0", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
    ])
    sa.remove_request("r1")
    assert sa.active_count() == 0
    assert sa.get_selected() == []


def test_local_sweep_improves():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42)
    sa.add_request("r1", [
        _b("b0", "r1", 10.0, {("A", "R"): 1}, {"A": 1}),
        _b("b1", "r1", 50.0, {("A", "R"): 1}, {"A": 1}),
    ])
    e_before = sa.get_energy()
    sa.local_sweep(n_steps=200, temperature=0.1)
    e_after = sa.get_energy()
    assert e_after <= e_before


def test_congestion_and_risk_params():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42,
                           congestion_weight=5.0, congestion_threshold=0.5,
                           risk_weight=2.0, risk_tau=5.0)
    sa.add_request("r1", [
        _b("b0", "r1", 50.0, {("A", "R"): 9}, {"A": 1}, lat=10.0),
    ])
    e = sa.get_energy()
    assert e > -50.0


def test_full_cooling_maintains_feasibility():
    ec = {("A", "R"): 2, ("R", "B"): 2}
    mc = {"A": 3, "R": 3, "B": 3}
    sa = StreamingAnnealer(ec, mc, seed=42)
    sa.add_request("r1", [
        _b("b0", "r1", 50.0, {("A", "R"): 1, ("R", "B"): 1}, {"A": 1, "R": 2, "B": 1}),
        _b("b1", "r1", 80.0, {("A", "R"): 3, ("R", "B"): 3}, {"A": 3, "R": 6, "B": 3}),
    ])
    sa.add_request("r2", [
        _b("b0", "r2", 40.0, {("A", "R"): 2, ("R", "B"): 2}, {"A": 2, "R": 4, "B": 2}),
    ])
    sa.full_cooling_cycle(max_iterations=500)
    sel = sa.get_selected()
    assert len(sel) >= 1


def test_history():
    ec, mc = _caps()
    sa = StreamingAnnealer(ec, mc, seed=42)
    sa.add_request("r1", [
        _b("b0", "r1", 10.0, {}, {}),
    ])
    assert len(sa.get_history()) >= 1
