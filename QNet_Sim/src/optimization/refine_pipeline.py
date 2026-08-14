from .tensor_network_optimizer import TensorNetworkOptimizer
from .metropolis_annealer import MetropolisAnnealer


class TensorAnnealerPipeline:
    """Run TN optimizer first, then refine with Metropolis annealing."""

    def __init__(self, bundles, edge_capacities, memory_capacities, seed=None):
        self.tn = TensorNetworkOptimizer(bundles, edge_capacities, memory_capacities)
        self.annealer = MetropolisAnnealer(bundles, edge_capacities, memory_capacities, seed=seed)

    def solve(self, edge_penalty=None, memory_penalty=None,
              congestion_penalty=0.05, memory_congestion_penalty=0.05,
              tn_bond_dim=8, tn_beta=5.0, tn_sweeps=15,
              anneal_max_iterations=2000, anneal_initial_temperature=3.0,
              anneal_cooling_rate=0.97, anneal_steps_per_temp=30,
              anneal_n_restarts=1):
        tn_result = self.tn.solve(
            edge_penalty=edge_penalty,
            memory_penalty=memory_penalty,
            congestion_penalty=congestion_penalty,
            memory_congestion_penalty=memory_congestion_penalty,
            bond_dim=tn_bond_dim,
            beta=tn_beta,
            max_sweeps=tn_sweeps,
        )
        tn_selections = tn_result["selections"]

        anneal_result = self.annealer.solve(
            edge_penalty=edge_penalty,
            memory_penalty=memory_penalty,
            congestion_penalty=congestion_penalty,
            memory_congestion_penalty=memory_congestion_penalty,
            max_iterations=anneal_max_iterations,
            initial_temperature=anneal_initial_temperature,
            cooling_rate=anneal_cooling_rate,
            steps_per_temperature=anneal_steps_per_temp,
            min_temperature=1e-3,
            patience=10,
            n_restarts=anneal_n_restarts,
            target_accept_rate=0.25,
            initial_selections=tn_selections,
        )
        return {
            "tn_energy": tn_result.get("energy", None),
            "final_energy": anneal_result["energy"],
            "selected": anneal_result["selected"],
            "selections": anneal_result["selections"],
        }
