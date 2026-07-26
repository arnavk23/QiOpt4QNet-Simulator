class MemoryDemandModel:
    @staticmethod
    def total_memory_demand(bell_pair_cost: int) -> int:
        """Return the total memory qubits required to create all Bell pairs in a bundle."""
        if bell_pair_cost < 0:
            raise ValueError("Bell-pair cost must be nonnegative")

        return 2 * bell_pair_cost
