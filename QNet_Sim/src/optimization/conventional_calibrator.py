def penalty_epsilon(p0):
    return max(1e-9, 1e-6*p0)
def conventional_coefficients(optimizer):
    p0=0.0
    for bundle in optimizer.bundles:
        p0=max(p0, bundle["utility"])
    penalty = p0 + penalty_epsilon(p0)
    return{"A":penalty, "B":penalty, "C":0.0, "D":penalty, "E":0.0}
