import simpy
import math
import pytest
from network.node import QuantumNode
from network.link import QuantumLink
from network.network import QuantumNetwork
from protocols.entanglement_generation import EntanglementGenerationProtocol

def test_node_decoherence():
    env = simpy.Environment()
    node = QuantumNode("Node_A", 10, t1=100.0, t2=50.0)
    
    # Reserve memory at t=0
    mem_ids = node.reserve_memory(1, env.now)
    assert len(mem_ids) == 1
    mem_id = mem_ids[0]
    
    # Initial fidelity should be 1.0
    assert node.calculate_fidelity(mem_id, env.now) == 1.0
    
    # Advance time by 10 units
    current_time = env.now + 10.0
    fidelity = node.calculate_fidelity(mem_id, current_time)
    
    # Expected: 1.0 * exp(-10/100) * exp(-10/50)
    expected_fidelity = math.exp(-10/100) * math.exp(-10/50)
    assert math.isclose(fidelity, expected_fidelity, rel_tol=1e-5)
    
def test_calculate_fidelity_uses_stored_state():
    import math as _math
    import numpy as np
    from models.quantum_state import QuantumState

    node = QuantumNode("Node_A", 10, t1=100.0, t2=50.0)
    mem_ids = node.reserve_memory(1, current_time=0.0)
    mem_id = mem_ids[0]

    # Werner state with parameter w=2/3 -> fidelity 0.75 with |Phi+>
    bell = np.array([1, 0, 0, 1]) / _math.sqrt(2)
    rho_bell = np.outer(bell, bell.conj())
    w = 2.0 / 3.0
    rho = w * rho_bell + (1 - w) * np.eye(4) / 4.0
    state = QuantumState(rho)
    assert _math.isclose(state.fidelity_with_bell(), 0.75, rel_tol=1e-9)

    node.assign_state(mem_id, state)
    # at creation time the stored pair's own fidelity is reported, not 1.0
    assert _math.isclose(node.calculate_fidelity(mem_id, 0.0), 0.75, rel_tol=1e-9)
    # after 10 time units the same base is decayed by the T1/T2 model
    expected = 0.75 * _math.exp(-10 / 100) * _math.exp(-10 / 50)
    assert _math.isclose(node.calculate_fidelity(mem_id, 10.0), expected, rel_tol=1e-9)


def test_release_memory_requires_explicit_eviction_policy():
    node = QuantumNode("Node_A", 10)
    node.reserve_memory(1, current_time=0.0)  # id 0
    node.reserve_memory(1, current_time=5.0)  # id 1

    # releasing by amount alone is ambiguous -> error, not a silent assumption
    with pytest.raises(ValueError):
        node.release_memory(amount=1)

    node.release_memory(amount=1, eviction="oldest")
    assert 0 not in node.memory_reservations
    assert 1 in node.memory_reservations

    node.release_memory(amount=1, eviction="newest")
    assert node.memory_used == 0
    assert node.memory_reservations == {}

    # releasing by explicit IDs remains unambiguous
    node.reserve_memory(2, current_time=1.0)
    node.release_memory(list(node.memory_reservations.keys()))
    assert node.memory_used == 0


def test_entanglement_generation_protocol():
    env = simpy.Environment()
    
    network = QuantumNetwork()
    node_a = QuantumNode("Node_A", 10)
    node_b = QuantumNode("Node_B", 10)
    network.add_node(node_a)
    network.add_node(node_b)
    
    # 100% success rate, latency of 5.0
    link = QuantumLink("Node_A", "Node_B", distance=10, generation_probability=1.0, raw_fidelity=0.9, latency=5.0, capacity=1)
    network.add_link(link)
    
    protocol = EntanglementGenerationProtocol(env, node_a, network, "Node_B")
    protocol.start()
    
    # Run simulation until there are no more events
    env.run()
    
    # Since probability is 1.0, it should succeed
    assert protocol.success is True
    # Should have taken exactly 5 units of time
    assert env.now == 5.0
    # Both nodes should have 1 memory used
    assert node_a.memory_used == 1
    assert node_b.memory_used == 1
