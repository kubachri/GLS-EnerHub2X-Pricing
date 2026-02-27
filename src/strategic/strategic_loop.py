# src/strategic/strategic_loop.py

import math
import re
import pandas as pd
from pyomo.environ import value, SolverFactory, TransformationFactory, Suffix, Objective, maximize, Var, NonNegativeReals, Binary
from pyomo.opt import TerminationCondition
from src.model.builder import build_model
from src.config import ModelConfig
import io
from contextlib import redirect_stdout
from src.strategic.submodel_biogas import build_biogas_model
from src.strategic.submodel_methanol import build_methanol_model


def run_cournot(cfg: ModelConfig, tol=1e-2, max_iter=50, damping=0):
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

    # Retrieve strategic fuel
    co2_label = cfg.co2_label

    # Retrieve strategic actors
    strategic_suppliers = ["Digester", "BiogasUpgrade", "Boiler", "CO2Liquefaction"]
    strategic_demanders = ["CO2Compressor", "CO2Storage", "MethanolSynthesis"]

    # Ensure strategic parameters are in the model
    if co2_label not in base_model.F:
        raise ValueError(f"Strategic fuel '{co2_label}' not found in model fuels. Available fuels: {base_model.F}")
    for tech in strategic_suppliers:
        if tech not in base_model.G:
            raise ValueError(f"Technology '{tech}' not found in data. Available technologies: {base_model.G}")
    for tech in strategic_demanders:    
        if tech not in base_model.G:
            if "storage" in tech.lower() and "storage" in cfg.data_file.lower():
                print(f"Note: Technology '{tech}' not found, but assumed to be voluntary (scenario specific).")
                strategic_demanders.remove(tech)
            else:
                raise ValueError(f"Technology '{tech}' not found in data. Available technologies: {base_model.G}")

    # -- Strategies initialization --
    # Initialize supply strategies = best-response to be iterated
    # Use Generation for CO2-producing techs as proxy for supply
    curr = {t:
            sum(
                value(base_model.Generation[tech, co2_label, t])
                for tech in strategic_suppliers 
                if (tech, co2_label) in base_model.f_out
                )
            for t in base_model.T
            } 

    total_supply = sum(curr[t] for t in curr)

    print(f"\nStrategic suppliers: {strategic_suppliers}")
    print(f"Initial strategy (total CO2 supply): {total_supply}\n")

    # Extract CO2 use (demand) and duals (willingness to pay) for curve construction
    strategic_demanders_without_storage = [tech for tech in strategic_demanders if "storage" not in tech.lower()]   # Avoid double-counting demand (the fuel can either be directly used or stored and then used - in the latter case, fuel use would be double-counted as demand)
    co2_use = {t: 
                sum(
                    value(base_model.Fueluse[tech, co2_label, t]) 
                    for tech in strategic_demanders_without_storage
                    if (tech, co2_label, t) in base_model.Fueluse
                    )
                for t in base_model.T 
               }
    co2_duals = {t: abs(base_model.dual.get(base_model.Balance['Skive', co2_label, t], 0.0)) for t in base_model.T} 

    print(f"Strategic demanders: {strategic_demanders}")
    print(f"CO2 initial demand: {sum(co2_use.values())}, average price: {sum(co2_duals.values())/len(co2_duals):.2f}")


    # -- Construct basis demand_price_blocks with dummy blocks (same for all t, initialized once) --

    # Case A: Realistic - competitive external CO2 demand on the market (observed by biogass)
    # Block 1: High price, low capacity
    # Block 2: Medium price, medium capacity
    # Block 3: Iteration-dependent, based on methanol CO2 demand
    # dummy_blocks = [
    #     {"block": 1, "price": 120.0, "capacity": 1.0},  # High price block
    #     {"block": 2, "price": 40.0, "capacity": 1.0}   # Medium price block
    # ]
    
    # Case B: NO external CO2 demand for biogas 
    # dummy_blocks = [] 

    # Case C: Hypothetical external CO2 demand for biogas - less competitive than methanol 
    # We do not want to use the external market to CREATE the competition but rather to cap it
    dummy_blocks = [
        {"block": 1, "price": 40.0, "capacity": 5.0},
    ]

    demand_price_blocks = {}
    for t in base_model.T:
        capacity = co2_use.get(t, 0.0)
        price = co2_duals[t] 
        block3 = {"block": 3, "price": price, "capacity": capacity}
        demand_price_blocks[t] = sorted(dummy_blocks + [block3], key=lambda x: -x['price'])
        for i, block in enumerate(demand_price_blocks[t]):
            block['block'] = i + 1  # Re-index blocks

    print(f"\nStrategies and demand curves initialized.\n")

    # Initialize dataframe to track iteration results if needed
    results_df = pd.DataFrame(columns=['Iteration', 'MaxChange', 'TotalCO2Supply', 'AvgCO2MarketPrice', 'AvailableCO2Supply', 'ObservedCO2Supply', 'TotalCO2Demand', 'AvgCO2Price', 'BiogasObj', 'MethanolObj'])
    results_df.set_index('Iteration', inplace=True)
    results_df.loc[0] = [None, total_supply, None, None, None, sum(co2_use.values()), sum(co2_duals.values())/len(co2_duals), None, None]

    duals_df = {}
    duals_df[0] = {t: co2_duals[t] for t in base_model.T}   

    # ------------------------------------------------------------ 
    # 2. BEST-RESPONSE ITERATION LOOP
    # ------------------------------------------------------------
    for iteration in range(1, max_iter+1):
        # print(f"----- Iteration {iteration} -----")
        max_change = 0.0

        # Optimize supply-side submodel (biogas)
        # Assuming only one strategic supplier for simplicity; extend as needed
        # print("Solving biogas submodel...")
        biogas_submodel = solve_biogas_submodel(cfg, demand_price_blocks, techs_biogas=strategic_suppliers)

        # Extract and update the supplier's best response
        co2_sell = {}
        co2_price = {}
        co2_supply = {}
        for t in biogas_submodel.T:
            # Extract optimal variables values
            co2_sell[t] = value(biogas_submodel.CO2_TotalSell[t]) if t in biogas_submodel.CO2_TotalSell else 0.0
            co2_price[t] = value(biogas_submodel.CO2_MarketPrice[t]) if t in biogas_submodel.CO2_MarketPrice else 0.0

            # Compute available CO2 supply for methanol submodel (how much of it is sold to methanol?) = state variable
            cumulative_higher_demand = sum(blk['capacity'] for blk in demand_price_blocks[t] if blk['price'] > co2_duals[t])
            co2_supply[t] = max(0.0, min(co2_sell[t] - cumulative_higher_demand, co2_use[t]))
            
            # Update current strategy with damping
            old_val = curr.get(t, 0.0)
            new_val = co2_supply[t]
            curr[t] = damping * old_val + (1 - damping) * new_val

            max_change = max(max_change, abs(curr[t] - old_val))

        # Print biogas submodel results
        # print(f"→ Total CO2 sold by biogas: {sum(co2_sell.values())}, including to methanol plant: {sum(co2_supply.values())}")

    
        # Optimize demand-side submodel (methanol) for next iteration: available CO2 supply is fixed (after damping)
        # print(f"Solving methanol submodel... (available supply from biogas = {sum(curr.values())})")
        methanol_submodel, co2_duals, co2_use = solve_methanol_submodel(cfg, co2_supply=curr, techs_methanol=strategic_demanders)

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
        results_df.loc[iteration] = [max_change,                               # Max change between iterations
                                    sum(co2_sell.values()),                    # Total CO2 supply from biogas
                                    sum(co2_price.values())/len(co2_price),    # Average CO2 market price for biogas
                                    sum(co2_supply.values()),                  # Available CO2 supply to methanol
                                    sum(curr.values()),                        # Observed CO2 supply for methanol (after damping)
                                    sum(co2_use.values()),                     # Total CO2 demand (internal)
                                    sum(co2_duals.values())/len(co2_duals),    # CO2 average price (dual)
                                    value(biogas_submodel.Obj),                # Biogas objective
                                    value(methanol_submodel.Obj)               # Methanol objective
                                    ]
        
        duals_df[iteration] = {t: co2_duals[t] for t in methanol_submodel.T}

        # Check convergence
        print(f"[ITER {iteration}] Max change = {max_change:.6f}")
        if max_change < tol:
            print(f"\n[INFO] Converged after {iteration} iterations (tol={tol}).\n")
            print("\n================= Cournot Loop Completed =================\n")
            break

    else:
        print("\n[WARN] Max iterations reached without full convergence.\n")
        # Find out a technique to force convergence if necessary or at least finalize differently

    # ------------------------------------------------------------
    # 3. FINALIZATION PHASE
    # ------------------------------------------------------------

    # Print iteration results summary
    print("\nCournot Iteration Results Summary:")
    print(results_df)

    duals_summary = []
    for t in methanol_submodel.T:
        row = {'Time': t}
        for iter in duals_df.keys():
            row[f'Dual_{iter}'] = duals_df[iter][t]
        duals_summary.append(row)
    duals_summary_df = pd.DataFrame(duals_summary)

    # Extract relevant duals
    duals = {f'{co2_label}': co2_duals}

    if hasattr(methanol_submodel, 'TargetDemand') and hasattr(methanol_submodel, 'DemandFuel'):
        for (step, af) in methanol_submodel.DemandFuel:
            constraint = methanol_submodel.TargetDemand[step, af]
            dual_val = methanol_submodel.dual.get(constraint, float("nan"))
            if duals.get(af) is None:
                duals[f'{af}'] = {}
            duals[f'{af}'][step] = dual_val

    # Objective decomposition for the final methanol submodel
    print("\nMethanol Submodel Objective Decomposition:")
    df_decomp_methanol = extract_objective_components(cfg, methanol_submodel, techs=strategic_demanders)
    print("\n", df_decomp_methanol)

    # Objective decomposition for the final biogas submodel
    print("\nBiogas Submodel Objective Decomposition:")
    df_decomp_biogas = extract_objective_components(cfg, biogas_submodel, techs=strategic_suppliers)
    print("\n", df_decomp_biogas)

    # Build and solve final full model with fixed strategies
    # final_model = solve_final_model(cfg, methanol_submodel, biogas_submodel, techs_methanol=strategic_demanders, techs_biogas=strategic_suppliers)

    return methanol_submodel, duals, [df_decomp_methanol, df_decomp_biogas, results_df, duals_summary_df]
    # return methanol_submodel, biogas_submodel, [results_df]


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


def solve_biogas_submodel(cfg, demand_price_blocks, techs_biogas):
    f = io.StringIO()
    with redirect_stdout(f):
        biogas_model = build_biogas_model(cfg, demand_price_blocks, techs_biogas)
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


def solve_methanol_submodel(cfg, co2_supply, techs_methanol):
    f = io.StringIO()
    with redirect_stdout(f):          # temporarily hide all print() inside
        methanol_submodel = build_methanol_model(cfg, co2_supply, techs_methanol)
        methanol_submodel.dual = Suffix(direction=Suffix.IMPORT)
        solver = SolverFactory('gurobi_persistent')
        solver.set_instance(methanol_submodel, symbolic_solver_labels=True)
        solver.options['MIPGap'] = 0.05
        mip_result = solver.solve(methanol_submodel, tee=True)

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

        # Extract CO2 use (demand for biogas) and duals (willingness to pay) for curve construction
        # co2_use = {t: 
        #             sum(
        #                 value(methanol_submodel.Fueluse[tech, co2_label, t]) 
        #                 for tech in techs_methanol
        #                 if (tech, co2_label, t) in methanol_submodel.Fueluse
        #                 )
        #             for t in methanol_submodel.T 
        #         }
        co2_use_internal = {t: value(methanol_submodel.CO2_InternalUse[t]) for t in methanol_submodel.T}
        co2_duals = {t: abs(methanol_submodel.dual.get(methanol_submodel.CO2_Balance[t], 0.0)) for t in methanol_submodel.T}

        # Print summary
        print(f"→ Total CO2 demand and average price: {sum(co2_use_internal.values())}, {sum(co2_duals.values())/len(co2_duals):.2f}")

    return methanol_submodel, co2_duals, co2_use_internal


def solve_final_model(cfg, methanol_submodel, biogas_submodel, techs_methanol, techs_biogas):
    """
    Solves the final full model with strategic variables fixed to their submodel solutions.
    
    Args:
        cfg: ModelConfig object
        methanol_submodel: Solved methanol submodel (provides fixed values for demanders)
        biogas_submodel: Solved biogas submodel (provides fixed values for suppliers)
        techs_methanol: List of strategic demander technologies
        techs_biogas: List of strategic supplier technologies
    
    Returns:
        final_model: The solved full model with fixed strategic variables
    """
    print("\n================= Building Final Full Model =================\n")
    
    # f = io.StringIO()
    # with redirect_stdout(f):

    # Build the full model
    final_model = build_model(cfg, final_strategic=True)
    final_model.dual = Suffix(direction=Suffix.IMPORT)
    
    # Fix generation variables for strategic suppliers to biogas submodel values
    print("Fixing strategic supplier variables...")
    for tech in techs_biogas:
        for f in final_model.F:
            if (tech, f) in final_model.f_out:
                for t in final_model.T:
                    if (tech, f, t) in biogas_submodel.Generation:
                        biogas_val = value(biogas_submodel.Generation[tech, f, t])
                        if (tech, f, t) in final_model.Generation:
                            final_model.Generation[tech, f, t].fix(biogas_val)
    
    # Fix fuel use variables for strategic demanders to methanol submodel values
    print("Fixing strategic demander variables...\n")
    for tech in techs_methanol:
        for f in final_model.F:
            if (tech, f) in final_model.f_out:
                for t in final_model.T:
                    if (tech, f, t) in methanol_submodel.Generation:
                        methanol_val = value(methanol_submodel.Generation[tech, f, t])
                        if (tech, f, t) in final_model.Generation:
                            final_model.Generation[tech, f, t].fix(methanol_val)
    
    # # Fix CO2 imports (from demander submodel) and exports (from supplier submodel) to ensure consistency
    # # Define prices
    # market_area = "DK1"
    # for t in final_model.T:
    #     final_model.price_buy[market_area, co2_label, t]= cfg.co2_market_price
    #     final_model.price_sale[market_area, co2_label, t]= 40.0  # corresponds to the price of the dummy block (additional demand from the external market)
    #     final_model.price_sale[market_area, co2_label, t]= value(biogas_submodel.CO2_MarketPrice[t])
    
    # # Fix variables
    # final_model.CO2_Buy  = Var(final_model.T, domain=NonNegativeReals)
    # final_model.CO2_Sale = Var(final_model.T, domain=NonNegativeReals)
    # for t in final_model.T:
    #     final_model.CO2_Buy[t].fix(value(methanol_submodel.Buy[market_area, co2_label, t]))
    #     final_model.CO2_Sale[t].fix(value(biogas_submodel.CO2_TotalSell[t]))
    
    # Solve the fixed model
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(final_model, symbolic_solver_labels=True)
    solver.options['MIPGap'] = 0.05
    mip_result = solver.solve(final_model, tee=True)
    
    inspect_model(final_model, solver, mip_result)
    
    return final_model

    


def safe_value(v, default=math.nan):
    try:
        return value(v)
    except ValueError:
        return default


from collections import defaultdict
def extract_objective_components(cfg, model, techs):
    decomp = []

    # Adapt to submodel objective structure (restricted to relevant sets since all techs are not optimized)
    fuels_in = set(f for (g,f) in model.f_in if g in techs)
    fuels_out = set(f for (g,f) in model.f_out if g in techs)
    DemandSet_restricted = [(a,e,t) for (a,e,t) in model.DemandSet if e in fuels_out]
    DemandFuel_restricted = [(s,f) for (s,f) in model.DemandFuel if f.split(".")[-1] in fuels_out]

    # Print check
    print("Restricted fuel sets (in; out):", fuels_in, fuels_out)
    print("Restricted DemandSet set:", DemandSet_restricted[:5], "... and values:", [model.demand[a,e,t] for (a,e,t) in DemandSet_restricted[:5]])
    print("Restricted DemandFuel set:", DemandFuel_restricted[:5], "... and values:", [model.DemandTarget[s,f] for (s,f) in DemandFuel_restricted[:5]])

    # a) Fuel imports (“Buy_…”) are costs → negative contributions
    for (a, e) in model.buyE:
        if e in fuels_in:
            tot = sum(
                safe_value(model.price_buy[a, e, t] * model.Buy[a, e, t])
                for t in model.T
            )
            decomp.append({
                "Element": f"Buy_{e}",
                "Contribution": - tot
            })

    # b) Fuel sales (“Sell_…”) are revenues → positive
    for (a, e) in model.saleE:
        if e in fuels_out:
            tot = sum(
                safe_value(model.price_sale[a, e, t] * model.Sale[a, e, t])
                for t in model.T
            )
            decomp.append({
                "Element": f"Sell_{e}",
                "Contribution": tot
            })

    # In the case of the supplier submodel, the strategic fuel sales need to be accounted
    if hasattr(model, "BLOCKS"):
        co2_sale_rev = sum(
            safe_value(model.Demand_BlockPrice[t, b] * (model.CO2_SellBlock[t, b] + model.CO2_ActiveBlock[t, b] * model.Demand_BlockCumCap[t, b]))
            for t in model.T
            for b in model.BLOCKS
            )
        decomp.append({
            "Element": f"Sell_{cfg.co2_label}",
            "Contribution": co2_sale_rev
        })

    # c) Variable O&M per technology
    varom_by_tech = defaultdict(float)
    for (g, e) in model.TechToEnergy:
        if g in techs and model.cvar[g] is not None:
            varom_by_tech[g] = sum(
                safe_value(model.Generation[g, e, t]) * model.cvar[g]
                for t in model.T
            )

    for g, val in varom_by_tech.items():
        decomp.append({
            "Element": f"Variable_OM_{g}",
            "Contribution": -val
        })

    # d) Startup costs
    tot_start = sum(
        safe_value(model.Startcost[g, t])
        for g in techs
        for t in model.T
    )
    decomp.append({"Element": "Startup Costs", "Contribution": - tot_start})

    # e) Slack penalties
    fuel_slack_totals = defaultdict(float)

    for (a, e, t) in DemandSet_restricted:
        var_import = model.SlackDemandImport[a, e, t]
        if var_import.value is not None and var_import.value != 0.0:
            fuel_slack_totals[f"Import_{e}"] += safe_value(var_import)
        
        var_export = model.SlackDemandExport[a, e, t]
        if var_export.value is not None and var_export.value != 0.0:
            fuel_slack_totals[f"Export_{e}"] += safe_value(var_export)

    for (step, af) in DemandFuel_restricted:
        var = model.SlackTarget[step, af]
        if var.value is not None:
            fuel_slack_totals[af] += safe_value(var)

    penalty = cfg.penalty

    for slack_type, slack_val in fuel_slack_totals.items():
        decomp.append({
            "Element": f"Slack {slack_type}",
            "Contribution": slack_val
        })
        decomp.append({
            "Element": f"Slack {slack_type} Cost",
            "Contribution": - penalty * slack_val
        })

    # f) Add TotalCost as the sum of all contributions
    total_profit = sum(entry["Contribution"] for entry in decomp if not math.isnan(entry["Contribution"]))
    decomp.append({
        "Element": "TotalProfit",
        "Contribution": total_profit
    })

    df_decomp = pd.DataFrame(decomp)
    return df_decomp

