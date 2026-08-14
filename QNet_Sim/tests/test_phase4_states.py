import numpy as np
import math
import simpy
import pytest

from models.quantum_state import QuantumState
from network.node import QuantumNode
from network.link import QuantumLink
from network.network import QuantumNetwork
from protocols.entanglement_swapping import EntanglementSwappingProtocol

def test_bell_state_fidelity():
    state = QuantumState.create_bell_state()
    assert math.isclose(state.fidelity_with_bell(), 1.0, rel_tol=1e-5)
    
def test_decoherence_kraus():
    state = QuantumState.create_bell_state()
    # Apply t1=10, t2=10, dt=10
    state.apply_decoherence(t1=10.0, t2=10.0, dt=10.0)
    
    # Fidelity should drop below 1.0
    f = state.fidelity_with_bell()
    assert f < 1.0
    assert f > 0.0
    
def test_quantum_state_validates_density_matrix():
    with pytest.raises(ValueError):
        QuantumState(np.zeros((2, 2)))  # wrong shape
    with pytest.raises(ValueError):
        QuantumState(np.eye(4) * 2)  # trace 8, not 1
    not_hermitian = np.array([[1, 1, 0, 0], [0, 0, 0, 0],
                              [0, 0, 0, 0], [0, 0, 0, 0]], dtype=complex)
    with pytest.raises(ValueError):
        QuantumState(not_hermitian)
    # a valid mixed (Werner-like) state passes and reports its fidelity
    bell = np.array([1, 0, 0, 1]) / math.sqrt(2)
    rho = 0.8 * np.outer(bell, bell.conj()) + 0.2 * np.eye(4) / 4
    s = QuantumState(rho)
    assert math.isclose(s.fidelity_with_bell(), 0.85, rel_tol=1e-9)


def test_apply_decoherence_validates_parameters():
    with pytest.raises(ValueError):
        QuantumState.create_bell_state().apply_decoherence(0.0, 50.0, 10.0)
    with pytest.raises(ValueError):
        QuantumState.create_bell_state().apply_decoherence(100.0, -5.0, 10.0)
    # dt <= 0 is a no-op; positive decay lowers fidelity
    s = QuantumState.create_bell_state()
    s.apply_decoherence(100.0, 50.0, 0.0)
    assert math.isclose(s.fidelity_with_bell(), 1.0, rel_tol=1e-9)
    s2 = QuantumState.create_bell_state()
    s2.apply_decoherence(100.0, 50.0, 10.0)
    assert s2.fidelity_with_bell() < 1.0


def test_entanglement_swap_perfect():
    state1 = QuantumState.create_bell_state()
    state2 = QuantumState.create_bell_state()
    
    swapped = QuantumState.entanglement_swap(state1, state2)
    assert math.isclose(swapped.fidelity_with_bell(), 1.0, rel_tol=1e-5)

def test_swapping_protocol():
    env = simpy.Environment()
    net = QuantumNetwork()
    
    node_a = QuantumNode("A", 10)
    node_b = QuantumNode("B", 10)
    node_c = QuantumNode("C", 10)
    
    net.add_node(node_a)
    net.add_node(node_b)
    net.add_node(node_c)
    
    net.add_link(QuantumLink("A", "B", 10, 1.0, 1.0, 5.0, 1))
    net.add_link(QuantumLink("B", "C", 10, 1.0, 1.0, 10.0, 1))
    
    # Pre-share entanglement
    mem_a = node_a.reserve_memory(1, env.now)[0]
    mem_b_a = node_b.reserve_memory(1, env.now)[0]
    
    mem_b_c = node_b.reserve_memory(1, env.now)[0]
    mem_c = node_c.reserve_memory(1, env.now)[0]
    
    state_ab = QuantumState.create_bell_state()
    node_a.assign_state(mem_a, state_ab)
    node_b.assign_state(mem_b_a, state_ab)
    
    state_bc = QuantumState.create_bell_state()
    node_b.assign_state(mem_b_c, state_bc)
    node_c.assign_state(mem_c, state_bc)
    
    protocol = EntanglementSwappingProtocol(
        env, node_b, net, "A", "C", mem_b_a, mem_b_c, mem_a, mem_c
    )
    protocol.start()
    
    env.run()
    
    # Time should be BSM time (1.0) + max latency (10.0) = 11.0
    assert env.now == 11.0
    assert protocol.success is True
    assert protocol.swapped_state is not None
    assert math.isclose(protocol.swapped_state.fidelity_with_bell(), 1.0, rel_tol=1e-5)
    
    # Node B should have released memory
    assert node_b.memory_used == 0
    assert node_a.memory_used == 1
    assert node_c.memory_used == 1
