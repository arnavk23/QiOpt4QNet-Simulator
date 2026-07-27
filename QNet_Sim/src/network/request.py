from itertools import count


class Request:
    _request_ids = count()

    def __init__(self, source: str, destination: str, minimum_fidelity: float, weight: float = 1.0,
                 request_id: str = None):
        self.request_id = request_id or f"request_{next(self._request_ids)}"
        self.source = source
        self.destination = destination
        self.minimum_fidelity = minimum_fidelity
        self.weight = weight
    
    def __repr__(self):
        return (f"Request({self.request_id}: {self.source} -> {self.destination}, "
                f"min_F = {self.minimum_fidelity}, weight = {self.weight})")
