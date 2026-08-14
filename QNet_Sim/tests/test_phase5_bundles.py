from network.network import QuantumNetwork
from network.link import QuantumLink
from network.node import QuantumNode
from network.request import Request
from routing.bundle_generation import BundleGenerator, Bundle

def test_bundle_generation():
    net = QuantumNetwork()
    net.add_node(QuantumNode("A", 10))
    net.add_node(QuantumNode("B", 10))
    net.add_node(QuantumNode("C", 10))
    
    # Links with low fidelity to ensure purification helps
    net.add_link(QuantumLink("A", "B", 10, 1.0, 0.8, 5.0, 1))
    net.add_link(QuantumLink("B", "C", 10, 1.0, 0.8, 5.0, 1))
    
    request = Request(source="A", destination="C", minimum_fidelity=0.5, weight=1.0)
    
    paths = [["A", "B", "C"]]
    
    generator = BundleGenerator(net)
    bundles = generator.generate_bundles(request, paths)
    
    # The higher cost/latency bundles should NOT be pruned if they offer higher fidelity
    # We should expect 3 bundles (q=0, q=1, q=2)
    assert len(bundles) == 3
    
    # Verify costs (q=0 -> 1 pair/link = 2. q=1 -> 2 pairs/link = 4. q=2 -> 4 pairs/link = 8)
    costs = [b.bell_pair_cost for b in bundles]
    assert sorted(costs) == [2, 4, 8]

    # Generated bundles retain the identifiers and core schema required by the
    # later optimizer stages.
    assert {b.request_id for b in bundles} == {request.request_id}
    assert len({b.bundle_id for b in bundles}) == len(bundles)
    for bundle in bundles:
        assert bundle.bundle_id.startswith(f"{request.request_id}_bundle_")
        assert bundle.path == paths[0]
        assert isinstance(bundle.memory_demand, int)
        assert isinstance(bundle.edge_demand, int)
        assert isinstance(bundle.utility, float)
        assert set(bundle.purification_profile) == {("A", "B"), ("B", "C")}
        assert bundle.edge_demands == {
            edge: 2 ** bundle.purification_rounds
            for edge in bundle.purification_profile
        }
        assert bundle.memory_demands == {
            "A": 2 ** bundle.purification_rounds,
            "B": 2 * 2 ** bundle.purification_rounds,
            "C": 2 ** bundle.purification_rounds,
        }

def test_sub_half_fidelity_links_skip_purification():
    net = QuantumNetwork()
    net.add_node(QuantumNode("A", 10))
    net.add_node(QuantumNode("B", 10))
    net.add_node(QuantumNode("C", 10))
    # Links below the 0.5 purification floor: purification rounds must be
    # skipped (not silently applied as no-ops), so fidelity stays flat.
    net.add_link(QuantumLink("A", "B", 10, 1.0, 0.4, 5.0, 1))
    net.add_link(QuantumLink("B", "C", 10, 1.0, 0.4, 5.0, 1))

    request = Request(source="A", destination="C", minimum_fidelity=0.2, weight=1.0)
    generator = BundleGenerator(net)
    bundles = generator.generate_bundles(request, [["A", "B", "C"]])

    # q=0, 1, 2 all survive the fidelity filter
    assert len(bundles) == 3
    fids = {b.purification_rounds: b.fidelity for b in bundles}
    # swap(0.4, 0.4) = 0.4*0.4 + (0.6*0.6)/3 = 0.28, identical at every depth
    assert all(abs(v - 0.28) < 1e-9 for v in fids.values())


def test_dominated_pruning():
    generator = BundleGenerator(QuantumNetwork())
    req = Request("A", "C", 0.5)
    
    # Bundle A: High fidelity, low cost (Dominates B)
    b_a = Bundle(req, ["A", "C"], 1, 0.9, 10.0, 2)
    # Bundle B: Low fidelity, high cost
    b_b = Bundle(req, ["A", "B", "C"], 0, 0.7, 20.0, 4)
    # Bundle C: Highest fidelity, high cost (Not dominated by A)
    b_c = Bundle(req, ["A", "D", "C"], 2, 0.95, 30.0, 8)
    
    pruned = generator.prune_dominated_bundles([b_a, b_b, b_c])
    
    # B should be pruned because A is strictly better in all metrics
    assert len(pruned) == 2
    assert b_a in pruned
    assert b_c in pruned
    assert b_b not in pruned
