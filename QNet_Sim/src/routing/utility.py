class UtilityModel:
    @staticmethod
    def calculate(
        request_weight: float,
        fidelity: float,
        min_required_fidelity: float,
        success_probability: float,
        latency: float,
        bell_pair_cost: int,
        lambda_latency: float = 1.0,
        lambda_cost: float = 1.0,
    ) -> float:
        """Utility defined in the QiOpt4QNet proposal."""

        fidelity_margin = max(0.0, fidelity - min_required_fidelity)

        return (
            request_weight
            * success_probability
            * (1.0 + fidelity_margin)
            - lambda_latency * latency
            - lambda_cost * bell_pair_cost
        )