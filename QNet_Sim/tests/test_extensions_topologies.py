"""Tests for the topology families (Extension 13)."""

import pytest

from extensions.topologies import (
    generate_ring_topology, generate_random_geometric_topology,
    generate_erdos_renyi_topology, generate_watts_strogatz_topology,
    generate_barabasi_albert_topology, all_topology_generators,
    topology_summary, topology_sweep, _is_connected,
)


def _schema_ok(topo):
    assert set(topo) >= {"nodes", "edges", "edge_capacities",
                         "memory_capacities", "link_params"}
    assert len(topo["edges"]) == len(topo["edge_capacities"])
    assert len(topo["nodes"]) == len(topo["memory_capacities"])
    for u, v in topo["edges"]:
        assert (u, v) in topo["link_params"] or (v, u) in topo["link_params"]
    assert _is_connected(topo["nodes"], topo["edges"])


@pytest.mark.parametrize("fn", [
    lambda: generate_ring_topology(n_nodes=8),
    lambda: generate_random_geometric_topology(n_nodes=12, radius=0.35),
    lambda: generate_erdos_renyi_topology(n_nodes=12, p=0.25),
    lambda: generate_watts_strogatz_topology(n_nodes=12, k_neighbors=2),
    lambda: generate_barabasi_albert_topology(n_nodes=12, m_links=2),
])
def test_topology_families_are_connected_and_schema_ok(fn):
    topo = fn()
    assert len(topo["nodes"]) == 12 or len(topo["nodes"]) == 8
    _schema_ok(topo)
    assert "topology_family" in topo


def test_ring_all_degree_two():
    topo = generate_ring_topology(n_nodes=6)
    degree = {n: 0 for n in topo["nodes"]}
    for u, v in topo["edges"]:
        degree[u] += 1
        degree[v] += 1
    assert all(d == 2 for d in degree.values())
    assert len(topo["edges"]) == 6  # n edges for n nodes (cycle)


def test_erdos_renyi_retries_until_connected():
    topo = generate_erdos_renyi_topology(n_nodes=10, p=0.3, seed=1)
    assert _is_connected(topo["nodes"], topo["edges"])


def test_random_geometric_has_positions():
    topo = generate_random_geometric_topology(n_nodes=10, radius=0.45, seed=2)
    assert len(topo["positions"]) == 10
    # every generated edge must be within radius
    for u, v in topo["edges"]:
        (x1, y1), (x2, y2) = topo["positions"][u], topo["positions"][v]
        assert ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 <= 0.45 + 1e-9


def test_topology_summary_fields():
    topo = generate_ring_topology(n_nodes=8)
    s = topology_summary(topo)
    assert s["n_nodes"] == 8 and s["n_edges"] == 8
    assert 0.0 < s["density"] <= 1.0
    assert s["min_degree"] == 2 and s["max_degree"] == 2
    assert s["mean_degree"] == pytest.approx(2.0)


def test_all_topology_generators_importable():
    gens = all_topology_generators()
    assert "chain" in gens and "ring" in gens and "barabasi_albert" in gens
    for name, fn in gens.items():
        topo = fn()
        assert _is_connected(topo["nodes"], topo["edges"]), name


def test_topology_sweep_rows():
    fns = {
        "ring": lambda: generate_ring_topology(n_nodes=8),
        "watts_strogatz": lambda: generate_watts_strogatz_topology(n_nodes=8,
                                                                   k_neighbors=2),
    }
    rows = topology_sweep(fns, n_requests=4, seed=42)
    assert len(rows) == 2
    for r in rows:
        assert r["topology"] in ("ring", "watts_strogatz")
        assert r["n_requests"] == 4
        assert 0 <= r["served_ratio"] <= 1
        assert "density" in r and "mean_degree" in r
