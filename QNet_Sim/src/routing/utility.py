class UtilityModel:
    @staticmethod
    def calculate(request_weight: float, fidelity: float, success_probability: float,
                  latency: float, bell_pair_cost: int) -> float:
        """Score a bundle by weighted successful fidelity per latency and Bell-pair cost."""
        if latency < 0:
            raise ValueError("Latency must be nonnegative")
        if bell_pair_cost < 0:
            raise ValueError("Bell-pair cost must be nonnegative")

        denominator = latency + bell_pair_cost
        if denominator == 0:
            return 0.0

        return request_weight * fidelity * success_probability / denominator
