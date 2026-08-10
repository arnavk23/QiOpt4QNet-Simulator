from .conventional_calibrator import penalty_epsilon

def possible_loads(key_demand_pairs, excluded_request):
    demands_via_request = {}
    for key, demand in key_demand_pairs:
        request_id = key[0]
        demands_via_request.setdefault(request_id,{0}).add(demand)
    loads={0}
    for request_id, demands in demands_via_request.items():
        if request_id==excluded_request:
            continue
        loads = {load+demand for load in loads for demand in demands}
    return loads

def coefficient_bound(grouped_demands, capacities, utilities):
    bound = 0.0
    for resource, key_demand_pairs in grouped_demands.items():
        capacity = capacities[resource]
        loads_by_request={}
        for key,demand in key_demand_pairs:
            request_id = key[0]
            if request_id not in loads_by_request:
                loads_by_request[request_id] = possible_loads(key_demand_pairs, request_id)
            loads = loads_by_request[request_id]

            violating = [load for load in loads if load+demand>capacity]
            if not violating:
                continue
            rho_asterik=min(violating)
            before = (rho_asterik+demand-capacity)**2
            after = max(rho_asterik-capacity,0)**2
            delta = before-after
            bound = max(bound, utilities[key]/delta)
    return bound

def proposed_global_coefficients(optimizer):
    utilities={}
    p0=0.0
    for bundle in optimizer.bundles:
        key=optimizer._bundle_key(bundle)
        utility = max(0.0,bundle["utility"])
        utilities[key]=utility
        p0=max(p0, utility)
    edge_bound = coefficient_bound(optimizer.edge_demands, optimizer.edge_capacities, utilities)
    memory_bound = coefficient_bound(optimizer.memory_demands, optimizer.memory_capacities, utilities)
    epsilon = penalty_epsilon(p0)
    return{
        "A":p0+epsilon,
        "B":edge_bound+epsilon,
        "D":memory_bound+epsilon,
        "C":0.0,
        "E":0.0
    }
#global coefficients for qubo
#keep A the same to see if change is due to the resource aware load consideration