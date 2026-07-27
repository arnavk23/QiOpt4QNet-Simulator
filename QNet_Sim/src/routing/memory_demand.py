from collections import defaultdict
from typing import Mapping


class MemoryDemandModel:
    @staticmethod
    def total_memory_demand(bell_pair_cost: int) -> int:
        """Return the total memory qubits required to create all Bell pairs in a bundle."""
        if not isinstance(bell_pair_cost, int) or isinstance(bell_pair_cost, bool) or bell_pair_cost < 0:
            raise ValueError("Bell-pair cost must be a nonnegative integer")

        return 2 * bell_pair_cost

    @staticmethod
    def per_node_memory_demand(edge_demands: Mapping[tuple[str, str], int]) -> dict[str, int]:
        """Return the memory qubits needed at each endpoint of the used edges."""
        memory_demands = defaultdict(int)
        for (source, destination), demand in edge_demands.items():
            if not isinstance(demand, int) or isinstance(demand, bool) or demand < 0:
                raise ValueError("Edge demand must be a nonnegative integer")
            memory_demands[source] += demand
            memory_demands[destination] += demand

        return dict(memory_demands)
