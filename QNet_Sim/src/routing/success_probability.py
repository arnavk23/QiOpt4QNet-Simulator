from typing import Sequence


class SuccessProbabilityModel:
    @staticmethod
    def purification_bbpssw_success(fidelity: float) -> float:
        """Return the success probability of one BBPSSW purification round."""
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError("Fidelity must be between 0 and 1")

        return (
            fidelity ** 2
            + 2.0 * fidelity * (1.0 - fidelity) / 3.0
            + 5.0 * ((1.0 - fidelity) / 3.0) ** 2
        )

    @classmethod
    def link_success_probability(cls, generation_probability: float, raw_fidelity: float,
                                 purification_rounds: int) -> float:
        """Return the probability that a purified Bell pair is available on one link."""
        if not 0.0 <= generation_probability <= 1.0:
            raise ValueError("Generation probability must be between 0 and 1")
        if purification_rounds < 0:
            raise ValueError("Purification rounds must be nonnegative")

        success_probability = generation_probability ** (2 ** purification_rounds)
        fidelity = raw_fidelity
        for round_index in range(purification_rounds):
            success_probability *= cls.purification_bbpssw_success(fidelity) ** (
                2 ** (purification_rounds - round_index - 1)
            )
            fidelity = cls._purified_fidelity(fidelity)

        return success_probability

    @staticmethod
    def path_success_probability(link_probabilities: Sequence[float]) -> float:
        """Return the probability that every link in a path succeeds."""
        success_probability = 1.0
        for probability in link_probabilities:
            success_probability *= probability
        return success_probability

    @staticmethod
    def _purified_fidelity(fidelity: float) -> float:
        if fidelity <= 0.5:
            return fidelity

        numerator = fidelity ** 2 + ((1.0 - fidelity) / 3.0) ** 2
        return numerator / SuccessProbabilityModel.purification_bbpssw_success(fidelity)
