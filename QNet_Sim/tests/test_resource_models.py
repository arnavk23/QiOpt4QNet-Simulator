import math

from routing.memory_demand import MemoryDemandModel
from routing.success_probability import SuccessProbabilityModel
from routing.utility import UtilityModel
from network.link import QuantumLink
from network.node import QuantumNode


def test_success_probability_accounts_for_generation_and_purification():
    fidelity = 0.8
    purification_success = (
        fidelity ** 2
        + 2.0 * fidelity * (1.0 - fidelity) / 3.0
        + 5.0 * ((1.0 - fidelity) / 3.0) ** 2
    )

    assert math.isclose(
        SuccessProbabilityModel.link_success_probability(0.9, fidelity, 0), 0.9
    )
    assert math.isclose(
        SuccessProbabilityModel.link_success_probability(0.9, fidelity, 1),
        0.9 ** 2 * purification_success,
    )
    assert math.isclose(SuccessProbabilityModel.path_success_probability([0.8, 0.5]), 0.4)


def test_memory_demand_is_two_qubits_per_bell_pair():
    assert MemoryDemandModel.total_memory_demand(3) == 6
    try:
        MemoryDemandModel.total_memory_demand(-1)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("Negative Bell-pair costs must be rejected")


def test_resource_demands_are_tracked_per_edge_and_node():
    edge_demands = {("A", "B"): 2, ("B", "C"): 1}

    assert MemoryDemandModel.per_node_memory_demand(edge_demands) == {
        "A": 2,
        "B": 3,
        "C": 1,
    }


def test_capacities_must_be_nonnegative_integers():
    QuantumLink("A", "B", 1.0, 1.0, 0.9, 1.0, 0)
    QuantumNode("A", 0)

    for capacity in (-1, 1.5):
        try:
            QuantumLink("A", "B", 1.0, 1.0, 0.9, 1.0, capacity)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid link capacity must be rejected")

        try:
            QuantumNode("A", capacity)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid memory capacity must be rejected")


def test_utility_matches_the_proposal_equation():
    utility = UtilityModel.calculate(
        request_weight=2.0,
        fidelity=0.9,
        min_required_fidelity=0.8,
        success_probability=0.5,
        latency=3.0,
        bell_pair_cost=4,
        lambda_latency=0.25,
        lambda_cost=0.1,
    )

    assert math.isclose(utility, 2.0 * 0.5 * 1.1 - 0.25 * 3.0 - 0.1 * 4)
