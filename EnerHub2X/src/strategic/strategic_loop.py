# src/strategic/strategic_loop.py

from pyomo.environ import value, SolverFactory, Suffix, Objective, maximize, Var
from src.model.builder import build_model
from src.config import ModelConfig
import io
from contextlib import redirect_stdout
from src.strategic.submodel_biogas import build_biogas_model
from src.strategic.submodel_methanol import build_methanol_model  # Assuming you implement this

from pyomo.environ import Var, Binary
from pyomo.environ import TransformationFactory
from pyomo.opt import TerminationCondition
import pandas as pd


def run_cournot(cfg: ModelConfig, tol=1e-3, max_iter=15, damping=0.1, co2_label='CO2'):
    """
    Iterative Cournot best-response algorithm for CO2 market.
    Each strategic supplier adjusts its sale quantity given competitors' fixed quantities.
    Convergence is declared when all strategy changes fall below `tol`.
    """
    print("\n================= Cournot Strategic Loop =================\n")

    # ------------------------------------------------------------
    # 1. INITIALIZATION PHASE
    # ------------------------------------------------------------
    print("Building and solving initial centralized model...")
    base_model = silent_build_model(cfg)  # Avoid prints

    print("Centralized baseline solve completed.")

    # Mapping tech -> area
    tech_to_area = {tech: area for (area, tech) in base_model.location}
    # area = tech_to_area.get(tech)

    # Retrieve strategic actors
    strategic_suppliers = ['BiogasUpgrade']
    strategic_demanders = ['CO2Compressor', 'CO2Storage', 'MethanolSynthesis']

    # Initialize supply strategies using CO2 generation (supply)
    curr = {} # supply current strategy
    for t in base_model.T:
        curr[t] = 0.0
        for tech in strategic_suppliers:
            # Use Generation for CO2-producing techs as proxy for sales
            gen_key = (tech, co2_label, t) if (tech, co2_label) in base_model.f_out else None
            curr[t] += value(base_model.Generation[gen_key]) if gen_key in base_model.Generation else 0.0
    total_supply = sum(curr[t] for t in curr)

    print(f"\nStrategic suppliers: {strategic_suppliers}")
    print(f"Initial strategy (total CO2 supply): {total_supply}\n")

    # Extract CO2 use (demand) and duals (willingness to pay) for curve construction
    co2_use = {t: 
                sum(
                    value(base_model.Fueluse[tech, co2_label, t]) 
                    for tech in strategic_demanders
                    if (tech, co2_label, t) in base_model.Fueluse
                    )
                for t in base_model.T 
               }
    co2_duals = {t: abs(base_model.dual.get(base_model.Balance['Skive', co2_label, t], 0.0)) for t in base_model.T} 

    print(f"Strategic demanders: {strategic_demanders}")
    print(f"CO2 initial demand: {sum(co2_use.values())}, average price: {sum(co2_duals.values())/len(co2_duals):.2f}")

    # Construct basis demand_price_blocks with dummy blocks (same for all t, initialized once)
    # Block 1: High price, low capacity
    # Block 2: Medium price, medium capacity
    # Block 3: Iteration-dependent, based on methanol CO2 demand
    dummy_blocks = [
        {"block": 1, "price": 120.0, "capacity": 1.0},  # High price block
        {"block": 2, "price": 70.0, "capacity": 1.0}   # Medium price block
    ]
    
    demand_price_blocks = {}
    for t in base_model.T:
        capacity = co2_use.get(t, 0.0)
        price = co2_duals[t] 
        block3 = {"block": 3, "price": price, "capacity": capacity}
        demand_price_blocks[t] = sorted(dummy_blocks + [block3], key=lambda x: -x['price'])
        for i, block in enumerate(demand_price_blocks[t]):
            block['block'] = i + 1  # Re-index blocks

    print(f"\nStrategies and demand curves initialized.")

    # Initialize dataframe to track iteration results if needed
    results_df = pd.DataFrame(columns=['Iteration', 'MaxChange', 'TotalCO2Supply', 'AvailableCO2Supply', 'TotalCO2Demand', 'AvgCO2Price', 'BiogasObj', 'MethanolObj'])
    results_df.set_index('Iteration', inplace=True)
    results_df.loc[0] = [None, total_supply, None, sum(co2_use.values()), sum(co2_duals.values())/len(co2_duals), None, None]

    # ------------------------------------------------------------ 
    # 2. BEST-RESPONSE ITERATION LOOP
    # ------------------------------------------------------------
    for iteration in range(1, max_iter+1):
        print(f"\n----- Iteration {iteration} -----\n")
        max_change = 0.0

        # Optimize supply-side submodel (biogas)
        # Assuming only one strategic supplier for simplicity; extend as needed
        print("Solving biogas submodel...")
        biogas_submodel = solve_biogas_submodel(cfg, demand_price_blocks)

        # Extract and update the supplier's best response
        co2_sell = {}
        co2_supply = {}
        for t in biogas_submodel.T:
            # Extract optimal variables values
            co2_sell[t] = value(biogas_submodel.CO2_TotalSell[t]) if t in biogas_submodel.CO2_TotalSell else 0.0
            co2_price = value(biogas_submodel.CO2_MarketPrice[t]) if t in biogas_submodel.CO2_MarketPrice else 0.0

            # Compute available CO2 supply for methanol submodel (how much of it is sold to methanol?) = state variable
            cumulative_higher_demand = sum(blk['capacity'] for blk in demand_price_blocks[t] if blk['price'] > co2_duals[t])
            co2_supply[t] = max(0.0, min(co2_sell[t] - cumulative_higher_demand, co2_use[t]))
            
            # Update current strategy with damping
            old_val = curr.get(t, 0.0)
            new_val = co2_supply[t]
            curr[t] = damping * new_val + (1 - damping) * old_val

            max_change = max(max_change, abs(curr[t] - old_val))

        # Check convergence
        print(f"\n[ITER {iteration}] Max change = {max_change:.6f}\n")
        if max_change < tol:
            print(f"\n[INFO] Converged after {iteration} iterations (tol={tol}).\n")
            print("\n================= Cournot Loop Completed =================\n")
            break
    
        # Optimize demand-side submodel (methanol) for next iteration: available CO2 supply is fixed (after damping)
        print(f"Solving methanol submodel... (available supply from biogas = {sum(curr.values())})")
        methanol_submodel, co2_duals, co2_use = solve_methanol_submodel(cfg, curr, demand_price_blocks)

        # Update demand_price_blocks for next iteration based on new CO2 demand
        demand_price_blocks = {}
        for t in methanol_submodel.T:
            capacity = co2_use.get(t, 0.0)
            price = co2_duals[t] 
            block3 = {"block": 3, "price": price, "capacity": capacity}
            demand_price_blocks[t] = sorted(dummy_blocks + [block3], key=lambda x: -x['price'])
            for i, block in enumerate(demand_price_blocks[t]):
                block['block'] = i + 1  # Re-index blocks

        # Log iteration results
        results_df.loc[iteration] = [max_change,                                # Max change between iterations
                                     sum(co2_sell.values()),                    # Total CO2 supply
                                     sum(co2_supply.values()),                  # Available CO2 supply to methanol
                                     sum(co2_use.values()),                     # Total CO2 demand
                                     sum(co2_duals.values())/len(co2_duals),    # CO2 average price
                                     value(biogas_submodel.Obj),                # Biogas objective
                                     value(methanol_submodel.Obj)               # Methanol objective
                                     ]

    else:
        print("\n[WARN] Max iterations reached without full convergence.\n")
        # Find out a technique to force convergence if necessary or at least finalize differently

    # ------------------------------------------------------------
    # 3. FINALIZATION PHASE
    # ------------------------------------------------------------

    # Print iteration results summary
    print("\nCournot Iteration Results Summary:")
    print(results_df)

    # Build and solve final full model with fixed strategies

    # Fix all strategic suppliers' sales

    # Compare results with centralized model
    
    return curr


# ============================================================
# Helper Functions
# ============================================================

def inspect_model(model, solver, result):

    # Get termination condition after solving the model
    term = result.solver.termination_condition

    if term == TerminationCondition.optimal:
        print("✔ Model solved to optimality.")

    else:
        print(f"→ Initial termination condition: {term}")

        # Handle the ambiguous case and retry
        if term == TerminationCondition.infeasibleOrUnbounded:
            print("⚠ Ambiguous (INF_OR_UNBD). Retrying with DualReductions=0 …")
            solver.options['DualReductions'] = 0
            solver.reset() # clear the persistent state
            retry_result = solver.solve(model, tee=True)
            term = retry_result.solver.termination_condition
            print(f"→ New termination condition: {term}")

        # Now term is either INFEASIBLE, UNBOUNDED, or OPTIMAL/OTHER
        if term == TerminationCondition.infeasible:
            print("✘ Model is infeasible. Extracting IIS …")
            grb = solver._solver_model
            grb.computeIIS()
            grb.write("model_iis.ilp")
            print(" → IIS written to model.ilp.iis.")
            return
        elif term == TerminationCondition.unbounded:
            print("⚠ MIP is unbounded (with integer vars).")
            return
        elif term == TerminationCondition.optimal:
            print("✔ Model solved to optimality.")
        else:
            return (f"‼️ Unexpected termination condition: {term}")
    
    # Now you know you have a valid solution
    mip_obj = value(model.Obj)
    print(f"✔ MIP objective = {mip_obj:,.2f}")

    return mip_obj

def silent_build_model(cfg):
    f = io.StringIO()
    with redirect_stdout(f):          # temporarily hide all print() inside
        model = build_model(cfg)
        model.dual = Suffix(direction=Suffix.IMPORT)
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(model, symbolic_solver_labels=True)
        solver.options['MIPGap'] = 0.05
        print("\nSolving MIP …\n")
        mip_result = solver.solve(model, tee=True)
        
        inspect_model(model, solver, mip_result)

        # After solving the MIP, but before fixing binaries:
        for v in model.component_data_objects(Var, descend_into=True):
            if v.domain is Binary and v.value is not None:
                v.fix(v.value)

        print("\nRelaxing integer vars → pure LP …\n")
        TransformationFactory('core.relax_integer_vars').apply_to(model)

        # Clear any old duals, then re‐solve as an LP to get duals
        print("Re-solving as an LP to extract duals …\n")
        lp_solver = SolverFactory('gurobi')
        lp_result = lp_solver.solve(model, tee=False, suffixes=['dual'])
        lp_obj = value(model.Obj)
        print(f"→ LP objective (continuous, binaries fixed) = {lp_obj:,.2f}\n")
        print("LP solve finished.\n")

    return model

def solve_biogas_submodel(cfg, demand_price_blocks):
    f = io.StringIO()
    with redirect_stdout(f):
        biogas_model = build_biogas_model(cfg, demand_price_blocks)
        biogas_model.dual = Suffix(direction=Suffix.IMPORT)
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(biogas_model, symbolic_solver_labels=True)
        solver.options['MIPGap'] = 0.05
        mip_result = solver.solve(biogas_model, tee=True)
        
    mip_obj = inspect_model(biogas_model, solver, mip_result)

    # # Assesment of variables when model is unbounded and artificially fixed
    # ARTIFICIAL_UB = 1e9
    # suspects = []

    # for v in biogas_model.component_objects(Var, active=True):
    #     for idx in v:
    #         var = v[idx]
    #         if var.is_binary():
    #             continue
    #         else:
    #             if var.ub is not None:
    #                 val = value(var, exception=False)
    #                 if val is not None and math.isfinite(val):
    #                     if abs(val - var.ub) <= max(1e-3, 1e-6 * var.ub):
    #                         suspects.append((var.name, idx, val, var.ub))

    # if suspects:
    #     print("Variables hitting upper bound:")
    #     for name, idx, val, ub in suspects:
    #         print(f"{name}{idx}: value={val}, ub={ub}")

    return biogas_model


def solve_methanol_submodel(cfg, co2_supply, demand_price_blocks, co2_label='CO2', strategic_demanders=["CO2Compressor", "CO2Storage", "MethanolSynthesis"]):
    f = io.StringIO()
    with redirect_stdout(f):          # temporarily hide all print() inside
        methanol_submodel = build_methanol_model(cfg, co2_supply, demand_price_blocks)
        methanol_submodel.dual = Suffix(direction=Suffix.IMPORT)
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(methanol_submodel, symbolic_solver_labels=True)
        solver.options['MIPGap'] = 0.05
        mip_result = solver.solve(methanol_submodel, tee=True)
        # solver.solve(methanol_submodel, tee=False)

    inspect_model(methanol_submodel, solver, mip_result)

    # After solving the MIP, but before fixing binaries:
    for v in methanol_submodel.component_data_objects(Var, descend_into=True):
        if v.domain is Binary and v.value is not None:
            v.fix(v.value)

    # Relax integer vars → pure LP
    TransformationFactory('core.relax_integer_vars').apply_to(methanol_submodel)

    # Clear any old duals, then re‐solve as an LP to get duals
    lp_solver = SolverFactory('gurobi')
    lp_result = lp_solver.solve(methanol_submodel, tee=False, suffixes=['dual'])
    lp_obj = value(methanol_submodel.Obj)
    print(f"→ LP objective (continuous, binaries fixed) = {lp_obj:,.2f}")

    # Extract CO2 use (demand) and duals (willingness to pay) for curve construction
    co2_use = {t: 
                sum(
                    value(methanol_submodel.Fueluse[tech, co2_label, t]) 
                    for tech in strategic_demanders
                    if (tech, co2_label, t) in methanol_submodel.Fueluse
                    )
                for t in methanol_submodel.T 
            }
    co2_duals = {t: abs(methanol_submodel.dual.get(methanol_submodel.CO2_Balance[t], 0.0)) for t in methanol_submodel.T}

    # Print summary
    print(f"→ Total CO2 demand and average price: {sum(co2_use.values())}, {sum(co2_duals.values())/len(co2_duals):.2f}")

    return methanol_submodel, co2_duals, co2_use
