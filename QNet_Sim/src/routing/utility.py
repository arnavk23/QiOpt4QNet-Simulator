class UtilityModel:
    @staticmethod
    def calculate(
        request_weight: float,
        fidelity: float,
        min_required_fidelity: float,
        success_probability: float,
        latency: float,
        bell_pair_cost: int,
        lambda_latency: float = 0.01,
        lambda_cost: float = 0.05,
        alpha_prob: float = 1.0,
        beta_fidelity: float = 1.0,
    ) -> float:
        """Utility defined in the QiOpt4QNet proposal."""

        fidelity_margin = max(0.0, fidelity - min_required_fidelity)

        return (
            request_weight
            * (alpha_prob * success_probability)
            * (1.0 + beta_fidelity * fidelity_margin)
            - lambda_latency * latency
            - lambda_cost * bell_pair_cost
        )