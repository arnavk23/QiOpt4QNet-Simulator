import simpy
import sys
import os

# Ensure src is in the path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, 'src'))
sys.path.insert(0, src_dir)

from network.node import QuantumNode
from network.link import QuantumLink
from network.network import QuantumNetwork
from protocols.entanglement_generation import EntanglementGenerationProtocol
from protocols.entanglement_swapping import EntanglementSwappingProtocol
from network.request import Request
from routing.bundle_generation import BundleGenerator

def get_float_input(prompt, default):
    val = input(f"{prompt} (default {default}): ").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print("Invalid input, using default.")
        return default

def get_int_input(prompt, default):
    val = input(f"{prompt} (default {default}): ").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        print("Invalid input, using default.")
        return default

def main():
    print("==================================================")
    print("  Interactive Quantum Network Simulation Demo     ")
    print("==================================================")
    
    print("\n--- Configure Nodes ---")
    t1 = get_float_input("T1 Coherence Time (ms)", 100.0)
    t2 = get_float_input("T2 Coherence Time (ms)", 50.0)
    mem_capacity = get_int_input("Memory Capacity per node", 10)
    
    print("\n--- Configure Links (Alice-Repeater-Bob) ---")
    dist_ab = get_float_input("Distance Alice-Repeater (km)", 10.0)
    dist_bc = get_float_input("Distance Repeater-Bob (km)", 15.0)
    raw_fidelity = get_float_input("Raw Link Fidelity (0.0 - 1.0)", 0.95)
    prob = get_float_input("Generation Probability (0.0 - 1.0)", 1.0)
    
    # Simple assumption: 1 km = 0.5 ms latency
    lat_ab = dist_ab * 0.5
    lat_bc = dist_bc * 0.5
    print(f"Calculated Latencies -> Alice-Repeater: {lat_ab}ms, Repeater-Bob: {lat_bc}ms")
    
    print("\n--- Configure Request ---")
    min_fid = get_float_input("Minimum End-to-End Fidelity required", 0.8)

    print("\n==================================================")
    print("                 STARTING SIMULATION              ")
    print("==================================================")
    
    env = simpy.Environment()
    net = QuantumNetwork()
    
    node_a = QuantumNode("Alice", memory_capacity=mem_capacity, t1=t1, t2=t2)
    node_b = QuantumNode("Repeater", memory_capacity=mem_capacity, t1=t1, t2=t2)
    node_c = QuantumNode("Bob", memory_capacity=mem_capacity, t1=t1, t2=t2)
    
    net.add_node(node_a)
    net.add_node(node_b)
    net.add_node(node_c)
    
    link_ab = QuantumLink("Alice", "Repeater", distance=dist_ab, generation_probability=prob, raw_fidelity=raw_fidelity, latency=lat_ab, capacity=1)
    link_bc = QuantumLink("Repeater", "Bob", distance=dist_bc, generation_probability=prob, raw_fidelity=raw_fidelity, latency=lat_bc, capacity=1)
    
    net.add_link(link_ab)
    net.add_link(link_bc)
    
    print("\n--- 1. Bundle Generation Demo ---")
    request = Request(source="Alice", destination="Bob", minimum_fidelity=min_fid, weight=1.0)
    paths = [["Alice", "Repeater", "Bob"]]
    
    generator = BundleGenerator(net)
    bundles = generator.generate_bundles(request, paths)
    
    print(f"Generated {len(bundles)} valid bundles for request Alice->Bob:")
    for idx, b in enumerate(bundles):
        print(f"  Bundle {idx}: Purification={b.purification_rounds}, Expected Fidelity={b.fidelity:.4f}, Cost={b.bell_pair_cost} pairs, Expected Latency={b.latency}ms")
    
    print(f"\n--- 2. SimPy Protocol Simulation ---")
    print(f"[Time: {env.now}] Starting protocols...")
    
    gen_protocol_ab = EntanglementGenerationProtocol(env, node_a, net, "Repeater")
    gen_protocol_bc = EntanglementGenerationProtocol(env, node_b, net, "Bob")
    
    gen_protocol_ab.start()
    gen_protocol_bc.start()
    
    # Run simulation enough for generation to finish
    max_lat = max(lat_ab, lat_bc)
    env.run(until=max_lat + 5.0)
    
    if not (gen_protocol_ab.success and gen_protocol_bc.success):
        print(f"\n[Time: {env.now}] Generation failed due to probabilistic link drops.")
        return
        
    print(f"\n[Time: {env.now}] Entanglement Generation Successful!")
    mem_a, mem_b_a = gen_protocol_ab.generated_memory_ids
    mem_b_c, mem_c = gen_protocol_bc.generated_memory_ids
    
    state_ab = node_a.get_state(mem_a)
    state_bc = node_b.get_state(mem_b_c)
    
    # Apply decoherence for the time they sat in memory
    dt_ab = env.now - lat_ab
    dt_bc = env.now - lat_bc
    state_ab.apply_decoherence(node_a.t1, node_a.t2, dt=dt_ab)
    state_bc.apply_decoherence(node_b.t1, node_b.t2, dt=dt_bc)
    
    print(f"Alice-Repeater Fidelity (Decohered for {dt_ab}ms): {state_ab.fidelity_with_bell():.4f}")
    print(f"Repeater-Bob Fidelity (Decohered for {dt_bc}ms): {state_bc.fidelity_with_bell():.4f}")
    
    print(f"\n[Time: {env.now}] Repeater is starting Entanglement Swapping...")
    swap_protocol = EntanglementSwappingProtocol(
        env, node_b, net, "Alice", "Bob", mem_b_a, mem_b_c, mem_a, mem_c
    )
    swap_protocol.start()
    
    env.run()
    
    if swap_protocol.success:
        print(f"\n[Time: {env.now}] Swapping Complete!")
        final_state = swap_protocol.swapped_state
        print(f"Final End-to-End Fidelity (Alice <---> Bob): {final_state.fidelity_with_bell():.4f}")
        print(f"Repeater memory used: {node_b.memory_used} (Should be 0, as it was traced out)")
        print(f"Alice memory used: {node_a.memory_used}")
        print(f"Bob memory used: {node_c.memory_used}")

if __name__ == "__main__":
    main()
