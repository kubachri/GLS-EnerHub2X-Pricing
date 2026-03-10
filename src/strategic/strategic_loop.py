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


def run_co2_market_equilibrium(cfg: ModelConfig, tol=1e-2, max_iter=30):
    """
    Iterative price-adjustment algorithm approximating a decentralized
    CO₂ market equilibrium between suppliers and demanders.
    Convergence is declared when all strategy changes fall below `tol`.
    """
    print("\n================= CO2 Market Equilibrium =================\n")

    # ------------------------------------------------------------
    # 1. INITIALIZATION PHASE
    # ------------------------------------------------------------
    print("Building and solving initial centralized model...")
    base_model = silent_build_model(cfg)  # Avoid prints

    print("Centralized baseline solve completed.")

    # Strategic parameters for iterative algorithm
    # alpha = 0.5  # Price adjustment step size (can be tuned for convergence behavior)
    alpha = [1]*10 + [0.8]*5 + [0.5]*5 + [0.3]*5 + [0.1]*15  # Decreasing step size over iterations

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
    co2_supply = {t:
            sum(
                value(base_model.Generation[tech, co2_label, t])
                for tech in strategic_suppliers 
                if (tech, co2_label) in base_model.f_out
                )
            for t in base_model.T
            } 
    total_supply = sum(co2_supply.values()) 

    print(f"\nStrategic suppliers: {strategic_suppliers}")
    print(f"Initial strategy (total CO2 supply): {sum(co2_supply.values()):.2f}\n")

    # Extract CO2 use (demand) and duals (willingness to pay) for curve construction
    co2_demand = co2_supply.copy()
    # co2_wtp = {t: abs(base_model.dual.get(base_model.Balance['Skive', co2_label, t], 0.0)) for t in base_model.T} 
    co2_wtp = {t: 150.0 for t in base_model.T}  # Start with high price to ensure initial demand is met

    print(f"Strategic demanders: {strategic_demanders}")
    print(f"CO2 initial demand: {sum(co2_demand.values()):.2f}, average price: {sum(co2_wtp.values())/len(co2_wtp):.2f}")

    # Define market scale for price adjustment (can be tuned for convergence behavior)
    market_scale = total_supply / base_model.T.__len__()

    # Initialize dataframe to track iteration results if needed
    results_df = pd.DataFrame(columns=['Iteration', 'PriceChange', 'QuantityChange', 'TotalCO2Supply', 'TotalCO2Demand', 'AvgExcessDemand', 'AvgCO2Price', 'BiogasObj', 'MethanolObj'])
    results_df.set_index('Iteration', inplace=True)
    results_df.loc[0] = [None, None, total_supply, total_supply, 0.0, sum(co2_wtp.values())/len(co2_wtp), None, None]

    bestResponse_df = {}
    bestResponse_df[0] = {t: (co2_supply[t], co2_demand[t], co2_wtp[t]) for t in base_model.T}   

    # ------------------------------------------------------------ 
    # 2. BEST-RESPONSE ITERATION LOOP
    # ------------------------------------------------------------
    # Assuming only one strategic supplier/demander for simplicity; extend as needed

    for iteration in range(1, max_iter+1):
        quantity_change = 0.0
        price_change = 0.0

        # Optimize demand-side submodel (methanol) 
        methanol_submodel = solve_methanol_submodel(cfg, price_co2_internal=co2_wtp, techs_methanol=strategic_demanders)

        # Extract and update the demander's best response
        co2_external = {t: value(methanol_submodel.Buy['DK1', co2_label, t]) for t in methanol_submodel.T if ('DK1', co2_label, t) in methanol_submodel.Buy}
        for t in methanol_submodel.T:
            old_val = co2_demand.get(t, 0.0)
            co2_demand[t] = value(methanol_submodel.CO2_InternalUse[t])

            # Compute maximal change for demand strategy to track convergence
            quantity_change = max(quantity_change, abs(co2_demand[t] - old_val))


        # Optimize supply-side submodel (biogas)
        biogas_submodel = solve_biogas_submodel(cfg, price_co2_internal=co2_wtp, techs_biogas=strategic_suppliers)

        # Extract and update the supplier's best response 
        for t in biogas_submodel.T:
            old_val = co2_supply.get(t, 0.0)
            co2_supply[t] = sum(
                value(biogas_submodel.Generation[tech, co2_label, t]) if (tech, co2_label, t) in biogas_submodel.Generation else 0.0
                for tech in strategic_suppliers
                )

            # Compute maximal change for supply strategy to track convergence
            quantity_change = max(quantity_change, abs(co2_supply[t] - old_val))


        # Update willingness to pay depending on the relative excess demand in the internal CO2 market (proportional tatonnement)
        for t in methanol_submodel.T:
            p_old = co2_wtp[t]
            if p_old == 0.0:
                p_old = 1.0  

            excess = co2_demand[t] - co2_supply[t]
            ratio = excess / market_scale
            p_delta = ratio * p_old
            # p_delta = alpha * excess
            print(f"Iter {iteration}, Time {t}: excess={excess:.2f}, ratio={ratio:.2f}, p_old={p_old:.2f}, p_delta={p_delta:.2f}")

            p_delta = max(min(p_delta, 10.0), -10.0)  # limit price changes to avoid oscillations
            p_new = p_old + alpha[iteration] * p_delta

            # Price smoothing to further dampen oscillations
            # p_new = 0.7 * p_new + 0.3 * p_old
            co2_wtp[t] = min(max(p_new, 1.0), 150.0)

            price_change = max(price_change, abs(co2_wtp[t] - p_old))

            # _, previous_demand, previous_wtp = bestResponse_df[iteration-1][t]
            # observed_supply = curr[t]

            # if iteration == 1:
            #     if observed_supply == previous_demand:  # WTP high enough to meet demand
            #         co2_wtp[t] = previous_wtp 
            #     elif observed_supply < previous_demand: # WTP not high enough to meet demand, increase price for next iteration
            #         co2_wtp[t] = 150.0
            #         co2_use[t] = previous_demand

            # else:
            #     if co2_use[t] == 0:
            #         co2_wtp[t] = 0.0
            #     elif observed_supply == previous_demand:
            #         co2_wtp[t] = previous_wtp  
            #     elif observed_supply < previous_demand:
            #         co2_wtp[t] = max(previous_wtp * 1.05, 150)  


        # Log iteration results
        results_df.loc[iteration] = [price_change, 
                                     quantity_change, 
                                     sum(co2_supply.values()), 
                                     sum(co2_demand.values()), 
                                     sum(co2_demand[t] - co2_supply[t] for t in methanol_submodel.T)/len(methanol_submodel.T), 
                                     sum(co2_wtp.values())/len(co2_wtp), 
                                     value(biogas_submodel.Obj), 
                                     value(methanol_submodel.Obj)]
        
        bestResponse_df[iteration] = {t: (co2_supply[t], co2_demand[t], co2_wtp[t]) for t in methanol_submodel.T}


        # Check convergence
        print(f"[ITER {iteration}] Max change = {price_change:.6f}")
        if price_change < tol*100 and quantity_change < tol:
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

    br_summary = []
    for t in methanol_submodel.T:
        row = {'Time': t}
        for iter in bestResponse_df.keys():
            supply_val, demand_val, wtp_val = bestResponse_df[iter][t]
            row[f'Supply_{iter}'] = supply_val
            row[f'Demand_{iter}'] = demand_val
            row[f'WTP_{iter}'] = wtp_val
        br_summary.append(row)
    br_summary_df = pd.DataFrame(br_summary)

    # Extract relevant duals
    duals = {f'{co2_label}': co2_wtp}

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

    return methanol_submodel, duals, [df_decomp_methanol, df_decomp_biogas, results_df, br_summary_df]
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


def solve_biogas_submodel(cfg, price_co2_internal, techs_biogas):
    f = io.StringIO()
    with redirect_stdout(f):
        biogas_model = build_biogas_model(cfg, price_co2_internal, techs_biogas)
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


def solve_methanol_submodel(cfg, price_co2_internal, techs_methanol):
    co2_label = cfg.co2_label

    f = io.StringIO()
    with redirect_stdout(f):          # temporarily hide all print() inside
        methanol_submodel = build_methanol_model(cfg, price_co2_internal, techs_methanol)
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

        # # Print summary
        # print(f"→ Total CO2 demand and average price: {sum(co2_use.values())}, {sum(co2_wtp.values())/len(co2_wtp):.2f}")

    return methanol_submodel


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

