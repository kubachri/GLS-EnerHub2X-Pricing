from pyomo.environ import value, Constraint

def detect_max_constraint_violation(model, threshold=1e-6, top_n=5):

    violations = []

    for c in model.component_data_objects(Constraint, active=True):
        try:
            body_val = value(c.body)
            lower = value(c.lower) if c.has_lb() else None
            upper = value(c.upper) if c.has_ub() else None

            violation = 0.0
            if lower is not None and body_val < lower - threshold:
                violation = lower - body_val
            if upper is not None and body_val > upper + threshold:
                violation = body_val - upper

            if violation > threshold:
                violations.append((violation, c.name, lower, body_val, upper))

        except:
            continue  # skip evaluable issues safely

    violations.sort(reverse=True)

    if violations:
        print(f"Detected {len(violations)} constraints exceeding {threshold} tolerance:")
        for v, name, lb, val, ub in violations[:top_n]:
            print(f"Violation: {v:.3e} | Constraint: {name} | Lower: {lb} | Value: {val} | Upper: {ub}")
    else:
        print(f"No constraint violations exceeding {threshold} tolerance detected.")
