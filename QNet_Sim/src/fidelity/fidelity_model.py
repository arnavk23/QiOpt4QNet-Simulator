class FidelityModel:
    
    @staticmethod
    def _validate_fidelity(f: float, name: str = "fidelity") -> None:
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1 (got {f})")

    @staticmethod
    def werner_parameter(f: float) -> float:
        FidelityModel._validate_fidelity(f)
        return (4 * f - 1) / 3.0

    @staticmethod
    def entanglement_swapping(f1: float, f2: float) -> float:
     
        # Theoretical formula for swapping two Werner states
        FidelityModel._validate_fidelity(f1, "f1")
        FidelityModel._validate_fidelity(f2, "f2")
        return f1 * f2 + ((1 - f1) * (1 - f2)) / 3.0

    @staticmethod
    def purification_bbpssw(f: float) -> float:
        """BBPSSW purification of a Werner-state fidelity ``f``.

        Raises ``ValueError`` if ``f`` is outside ``[0, 1]`` or strictly below
        ``0.5``: the BBPSSW map is only improving above the 0.5 threshold
        (below it the formula is contracting), so a sub-0.5 input signals a
        caller bug or an infeasible pair rather than a silent no-op.  At
        exactly ``f = 0.5`` the map is the identity and returns ``0.5``.
        Callers that may hold sub-0.5 pairs must skip purification explicitly.
        """
        FidelityModel._validate_fidelity(f)
        if f < 0.5:
            raise ValueError(
                f"purification requires fidelity > 0.5 (got {f}); a sub-0.5 "
                "pair cannot be purified -- skip purification or reject it"
            )

        # BBPSSW analytic formula
        term1 = f**2 + ((1 - f) / 3.0)**2
        term2 = f**2 + (2.0 * f * (1 - f) / 3.0) + 5.0 * ((1 - f) / 3.0)**2
        return term1 / term2
        
    @staticmethod
    def end_to_end_fidelity(link_fidelities: list[float]) -> float:
       
        if not link_fidelities:
            return 0.0
        for f in link_fidelities:
            FidelityModel._validate_fidelity(f)
            
        final_f = link_fidelities[0]
        for next_f in link_fidelities[1:]:
            final_f = FidelityModel.entanglement_swapping(final_f, next_f)
            
        return final_f

    