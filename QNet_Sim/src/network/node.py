import math
from typing import Optional
from models.quantum_state import QuantumState

class QuantumNode:
    def __init__(self, node_id: str, memory_capacity: int, t1: float = float('inf'), t2: float = float('inf')):
        if not isinstance(memory_capacity, int) or isinstance(memory_capacity, bool) or memory_capacity < 0:
            raise ValueError("Memory capacity must be a nonnegative integer")
        self.id = node_id
        self.memory_capacity = memory_capacity
        self.t1 = t1
        self.t2 = t2
        
        # We still keep memory_used for backward compatibility with simple tests,
        # but we also introduce a dict to track details of individual memory qubits.
        self._memory_used_count = 0
        # memory_reservations[id] = {'creation_time': float, 'state': Optional[QuantumState]}
        self.memory_reservations = {}
        self._next_mem_id = 0
    
    @property
    def memory_used(self) -> int:
        return self._memory_used_count
        
    @memory_used.setter
    def memory_used(self, value: int):
        self._memory_used_count = value
    
    def reserve_memory(self, amount: int = 1, current_time: float = 0.0) -> list[int]:
        """Reserves memory and returns a list of memory IDs. Returns empty list if not enough memory."""
        if self.available_memory() >= amount:
            self._memory_used_count += amount
            reserved_ids = []
            for _ in range(amount):
                mem_id = self._next_mem_id
                self.memory_reservations[mem_id] = {'creation_time': current_time, 'state': None}
                reserved_ids.append(mem_id)
                self._next_mem_id += 1
            return reserved_ids
        return []
        
    def assign_state(self, mem_id: int, state: QuantumState):
        """Assigns a generated QuantumState to a reserved memory slot."""
        if mem_id in self.memory_reservations:
            self.memory_reservations[mem_id]['state'] = state
            
    def get_state(self, mem_id: int) -> Optional[QuantumState]:
        if mem_id in self.memory_reservations:
            return self.memory_reservations[mem_id]['state']
        return None

    def release_memory(self, memory_ids: Optional[list[int]] = None, amount: int = 1,
                       eviction: Optional[str] = None):
        """Release specific memory IDs, or ``amount`` slots under an explicit policy.

        Releasing by explicit ``memory_ids`` is unambiguous and preferred.
        Releasing by ``amount`` alone is ambiguous (which slots to evict is a
        caller decision), so it requires an explicit ``eviction`` policy:
        ``"oldest"`` or ``"newest"`` by creation time.  Passing neither
        raises instead of silently assuming an eviction order.
        """
        if memory_ids is not None:
            for mem_id in memory_ids:
                if mem_id in self.memory_reservations:
                    del self.memory_reservations[mem_id]
                    self._memory_used_count -= 1
            return

        if eviction not in ("oldest", "newest"):
            raise ValueError(
                "release_memory without explicit memory_ids is ambiguous: pass "
                "memory_ids, or an explicit eviction policy ('oldest' or "
                "'newest', by creation time)."
            )
        if self._memory_used_count < amount:
            self._memory_used_count = 0
            self.memory_reservations.clear()
            return
        self._memory_used_count -= amount
        order = sorted(self.memory_reservations.keys(),
                       key=lambda k: self.memory_reservations[k]['creation_time'],
                       reverse=(eviction == "newest"))
        for mem_id in order[:amount]:
            del self.memory_reservations[mem_id]

    def available_memory(self) -> int:
        return self.memory_capacity - self._memory_used_count

    def calculate_fidelity(self, mem_id: int, current_time: float, initial_fidelity: float = 1.0) -> float:
        """Fidelity of the stored pair at ``current_time`` under T1/T2 decoherence.

        The base fidelity is taken from the stored density matrix when one has
        been assigned (``QuantumState.fidelity_with_bell()``); otherwise it
        falls back to ``initial_fidelity``.  The base is then decayed by the
        scalar model ``exp(-dt/T1) * exp(-dt/T2)``.  The stored state is never
        mutated, so repeated calls at the same ``current_time`` agree.
        """
        if mem_id not in self.memory_reservations:
            return 0.0

        creation_time = self.memory_reservations[mem_id]['creation_time']
        state = self.memory_reservations[mem_id]['state']

        dt = current_time - creation_time
        base = state.fidelity_with_bell() if state is not None else initial_fidelity
        if dt <= 0:
            return base

        decay = math.exp(-dt / self.t1) * math.exp(-dt / self.t2)
        return base * decay

    def __repr__(self):
        return f"QuantumNode(id={self.id}, capacity={self.memory_capacity}, available={self.available_memory()}, T1={self.t1}, T2={self.t2})"

