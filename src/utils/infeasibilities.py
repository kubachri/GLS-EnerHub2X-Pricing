
def compute_gurobi_iis(model, solver):
    """
    Computes and extracts the IIS using Gurobi directly from Pyomo.
    Requires that the Pyomo solver is 'gurobi_persistent' for direct model access.
    """
    if solver._solver_model is None:
        print("Gurobi persistent model not available; cannot compute IIS.")
        return

    gurobi_model = solver._solver_model

    print("\nSolver reported infeasibility. Computing IIS with Gurobi...")
    gurobi_model.computeIIS()
    iis_filename = "model_iis.ilp"
    gurobi_model.write(iis_filename)
    print(f"IIS written to {iis_filename}. You can inspect it with Gurobi CLI or Gurobi Optimizer for details.\n")

    # Additionally, print a readable summary of constraints and bounds in the IIS:
    for c in gurobi_model.getConstrs():
        if c.IISConstr:
            print(f"Constraint in IIS: {c.ConstrName}")
    for v in gurobi_model.getVars():
        if v.IISLB:
            print(f"Variable lower bound in IIS: {v.VarName}")
        if v.IISUB:
            print(f"Variable upper bound in IIS: {v.VarName}")
    print("\nFinished IIS extraction.\n")
