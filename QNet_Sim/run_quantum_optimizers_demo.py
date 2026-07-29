import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from network.network import QuantumNetwork
from network.link import QuantumLink
from network.node import QuantumNode
from network.request import Request
from routing.bundle_generation import BundleGenerator
from optimization.metropolis_annealer import MetropolisAnnealer
from optimization.tensor_network_optimizer import TensorNetworkOptimizer


def main():
    print("=" * 60)
    print("   Quantum-Inspired Optimizers Demo")
    print("   (arXiv:2605.27425 — Metropolis & Tensor-Network)")
    print("=" * 60)

    net = QuantumNetwork()
    net.add_node(QuantumNode("Alice", memory_capacity=10))
    net.add_node(QuantumNode("Repeater", memory_capacity=10))
    net.add_node(QuantumNode("Bob", memory_capacity=10))

    net.add_link(QuantumLink("Alice", "Repeater", 10, 1.0, 0.85, 5.0, 6))
    net.add_link(QuantumLink("Repeater", "Bob", 10, 1.0, 0.85, 5.0, 6))

    req = Request("Alice", "Bob", minimum_fidelity=0.6, weight=50.0, request_id="Req_1")
    gen = BundleGenerator(net)
    bundles = gen.generate_bundles(req, [["Alice", "Repeater", "Bob"]])

    print(f"\nGenerated {len(bundles)} bundles for {req.request_id}")
    for b in bundles:
        print(f"  [{b.bundle_id}] q={b.purification_rounds}, F={b.fidelity:.4f}, "
              f"cost={b.bell_pair_cost}, U={b.utility:.4f}")

    opt_bundles = [b.to_optimizer_dict() for b in bundles]
    edge_caps = {("Alice", "Repeater"): 6, ("Repeater", "Bob"): 6}
    mem_caps = {"Alice": 10, "Repeater": 10, "Bob": 10}

    print("\n--- Optimizer A: Metropolis Annealer ---")
    annealer = MetropolisAnnealer(opt_bundles, edge_caps, mem_caps, seed=42)
    result_a = annealer.solve(
        penalty=100.0, edge_penalty=10.0, memory_penalty=10.0,
        max_iterations=2000, initial_temperature=10.0, cooling_rate=0.97
    )
    print(f"  Selected bundles: {result_a['selected']}")
    print(f"  Final energy: {result_a['energy']:.4f}")

    print("\n--- Optimizer B: Tensor-Network Compressor ---")
    tn_opt = TensorNetworkOptimizer(opt_bundles, edge_caps, mem_caps)
    result_b = tn_opt.solve(edge_penalty=10.0, memory_penalty=10.0, bond_dim=4, beta=5.0)
    print(f"  Selected bundles: {result_b['selected']}")

    print("\n" + "=" * 60)
    print("   Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
