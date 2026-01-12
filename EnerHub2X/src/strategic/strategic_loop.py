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
import math


def run_cournot(cfg: ModelConfig, tol=1e-3, max_iter=30, damping=0.6, co2_label='CO2'):
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

    # Retrieve strategic actors (assume list of techs, e.g., ['BiogasUpgrade'])
    # strategic_suppliers = getattr(base_model, 'StrategicSuppliers', ['BiogasUpgrade'])
    # strategic_demanders = getattr(base_model, 'StrategicDemanders', ['CO2Compressor', 'CO2Storage', 'MethanolSynthesis'])  
    strategic_suppliers = ['BiogasUpgrade']
    strategic_demanders = ['CO2Compressor', 'CO2Storage', 'MethanolSynthesis']

    # Initialize supply strategies using CO2 generation (supply)
    # curr = _initialize_strategies(base_model, strategic_suppliers, tech_to_area, co2_label)
    curr = {} # supply current strategy
    for tech in strategic_suppliers:
        curr[tech] = {}
        for t in base_model.T:
            # Use Generation for CO2-producing techs as proxy for sales
            gen_key = (tech, co2_label, t) if (tech, co2_label) in base_model.f_out else None
            curr[tech][t] = value(base_model.Generation[gen_key]) if gen_key in base_model.Generation else 0.0

    print(f"\nStrategic suppliers: {strategic_suppliers}")
    print(f"Initial strategy (total CO2 supply): {sum(curr[tech][t] for tech in curr for t in curr[tech])}")

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
        {"block": 1, "price": 100.0, "capacity": 50.0},  # High price block
        {"block": 2, "price": 50.0, "capacity": 150.0}   # Medium price block
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

        # _fix_competitor_sales(biogas_submodel, tech, strategic_suppliers, tech_to_area, curr, co2_label)

        # Extract and update the supplier's best response
        for t in biogas_submodel.T:

            # co2_sell = value(biogas_submodel.CO2_TotalSell[t]) if t in biogas_submodel.CO2_TotalSell else 0.0
            # co2_price = value(biogas_submodel.CO2_MarketPrice[t]) if t in biogas_submodel.CO2_MarketPrice else 0.0

            co2_sell = 0.0
            for tech in strategic_suppliers:
                # Use Generation for CO2-producing techs as proxy for sales
                gen_key = (tech, co2_label, t) if (tech, co2_label) in base_model.f_out else None
                new_val = value(biogas_submodel.Generation[gen_key]) if gen_key in base_model.Generation else 0.0
                old_val = curr[tech].get(t, 0.0)

                updated = damping * new_val + (1 - damping) * old_val
                curr[tech][t] = updated

                max_change = max(max_change, abs(new_val - old_val))

                co2_sell += new_val

            if co2_sell != value(biogas_submodel.CO2_TotalSell[t]):
                print(f"[WARN] Mismatch in total CO2 sell at time {t}: computed={co2_sell}, model={value(biogas_submodel.CO2_TotalSell[t])}")

        # Check convergence
        print(f"\n[ITER {iteration}] Max change = {max_change:.6f}\n")
        if max_change < tol:
            print(f"\n[INFO] Converged after {iteration} iterations (tol={tol}).\n")
            print("\n================= Cournot Loop Completed =================\n")
            break

        # Optimize demand-side submodel (methanol) for next iteration
        print("Solving methanol submodel...")
        methanol_submodel = build_methanol_model(cfg, curr, demand_price_blocks)
        methanol_submodel.dual = Suffix(direction=Suffix.IMPORT)
        solver = SolverFactory('gurobi_persistent')
        solver.solve(methanol_submodel, tee=False)

        demand_price_blocks = {}
        # Update demand_price_blocks for next iteration based on new CO2 demand
        for t in methanol_submodel.T:
            capacity = value(methanol_submodel.Fueluse['MethanolSynthesis', co2_label, t]) if ('MethanolSynthesis', co2_label, t) in methanol_submodel.Fueluse else 0.0
            price = abs(methanol_submodel.dual.get(methanol_submodel.Balance['Skive', 'CO2', t], 0.0)) if ('Skive', 'CO2', t) in methanol_submodel.Balance.index_set() else 0.0
            block3 = {"block": 3, "price": price, "capacity": capacity}
            demand_price_blocks[t] = sorted(dummy_blocks + [block3], key=lambda x: -x['price'])
            for i, block in enumerate(demand_price_blocks[t]):
                block['block'] = i + 1  # Re-index blocks

    else:
        print("[WARN] Max iterations reached without full convergence.\n")
        # Find out a technique to force convergence if necessary or at least finalize differently

    # ------------------------------------------------------------
    # 3. FINALIZATION PHASE
    # ------------------------------------------------------------
    # print("Building final model with fixed strategic sales...")
    # final_model = silent_build_model(cfg)
    # final_model.dual = Suffix(direction=Suffix.IMPORT)

    # # Fix all strategic suppliers' sales
    # _fix_all_sales(final_model, strategic_suppliers, tech_to_area, curr, co2_label)

    # print("\nSolving final full model (MIP)...\n")
    # final_solver = SolverFactory('gurobi_persistent')
    # final_solver.set_instance(final_model, symbolic_solver_labels=True)
    # final_solver.options['MIPGap'] = 0.05
    # final_solver.solve(final_model, tee=True)

    # Compare results with centralized model
    print("\n[CHECK] Comparing centralized vs strategic results...")

    # # Central model
    # q_central = {}
    # for (area, fuel) in base_model.saleE:
    #     if fuel == co2_label:
    #         q_central[(area,t)] = value(base_model.Sale[area,fuel,t])

    # # Strategic model
    # q_strat = {}
    # for (area, fuel) in final_model.saleE:
    #     if fuel == co2_label:
    #         q_strat[(area,t)] = value(final_model.Sale[area,fuel,t])

    # if not q_central:
    #     print("[WARN] No CO2 sales found in centralized model for comparison.")
    # for key in q_central:
    #     print(f"  {key}: central={q_central[key]:.3f}, strategic={q_strat[key]:.3f}, Δ={q_strat[key]-q_central[key]:.3e}")
    
    return curr


# ============================================================
# Helper Functions
# ============================================================

def _initialize_strategies(model, suppliers, tech_to_area, co2_label):
    """Extract initial sale quantities for each strategic supplier."""
    curr = {}
    for tech in suppliers:
        area = tech_to_area.get(tech)
        curr[tech] = {}
        for t in model.T:
            idx = (area, co2_label)
            curr[tech][t] = value(model.Sale[area, co2_label, t]) if idx in model.saleE else 0.0
    return curr


def _fix_competitor_sales(model, current_tech, suppliers, tech_to_area, curr, co2_label):
    """Fix competitors' sale variables to current strategy values."""
    for other in suppliers:
        if other == current_tech:
            continue
        area_other = tech_to_area.get(other)
        if area_other is None:
            continue
        for t in model.T:
            idx = (area_other, co2_label)
            if idx in model.saleE:
                val = curr[other].get(t, 0.0)
                model.Sale[area_other, co2_label, t].fix(val)
                

def _define_strategic_objective(model, current_tech, suppliers, tech_to_area, curr, co2_label):
    """Define the profit-maximizing objective for a strategic supplier."""
    # Build competitor sales sum for each timestep
    comp_sales = {t: sum(curr[other].get(t, 0.0)
                        for other in suppliers if other != current_tech)
                for t in model.T}

    # Define temporary profit objective
    profit_expr = 0
    area = tech_to_area.get(current_tech)
    for t in model.T:
        idx = (area, co2_label)
        if idx not in model.saleE:
            continue
        sale_var = model.Sale[area, co2_label, t]
        a = value(model.a_co2[t])
        b = value(model.b_co2[t])
        # inverse demand: p = a - b*(q_i + q_-i)
        price_expr = a - b * (comp_sales[t] + sale_var)
        profit_expr += price_expr * sale_var
        # optional: subtract linear cost approximation if available
        # e.g., profit_expr -= m.cvar[tech] * sale_var  # if m.cvar exists

    # Deactivate existing objective (keep reference if needed)
    for obj in list(model.component_data_objects(Objective, active=True)):
        obj.deactivate()

    model.ProfitObj = Objective(expr=profit_expr, sense=maximize)
    print(f"[DEBUG] Profit objective set for {current_tech}")


def _update_strategy(model, tech, tech_to_area, curr, co2_label, damping):
    """Extract new sale values for a supplier and apply damping update."""
    area = tech_to_area.get(tech)
    max_change = 0.0
    for t in model.T:
        idx = (area, co2_label)
        if idx not in model.saleE:
            # print(f"[WARN] No sale variable for {tech} at time {t}. Skipping update.")
            continue
        new_val = value(model.Sale[area, co2_label, t])
        old_val = curr[tech].get(t, 0.0)
        updated = damping * new_val + (1 - damping) * old_val
        curr[tech][t] = updated
        max_change = max(max_change, abs(updated - old_val))
    return max_change


def _fix_all_sales(model, suppliers, tech_to_area, curr, co2_label):
    """Fix all strategic suppliers' sales in the final model."""
    for tech in suppliers:
        area = tech_to_area.get(tech)
        if area is None:
            continue
        for t in model.T:
            idx = (area, co2_label)
            if idx in model.saleE:
                model.Sale[area, co2_label, t].fix(curr[tech][t])


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
        term = mip_result.solver.termination_condition
        print(f"\n→ Initial termination condition: {term}")

        # After solving the MIP, but before fixing binaries:
        for v in model.component_data_objects(Var, descend_into=True):
            if v.domain is Binary and v.value is not None:
                v.fix(v.value)

        print("\nRelaxing integer vars → pure LP …\n")
        TransformationFactory('core.relax_integer_vars').apply_to(model)

        # Clear any old duals, then re‐solve as an LP to get duals
        print("Re‐solving as an LP to extract duals …\n")
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
        term = mip_result.solver.termination_condition

    # print(f"\n→ Initial termination condition: {term}")
    # print("\nMIP solve finished.\n")

    # Handle the ambiguous case and retry
    if term == TerminationCondition.infeasibleOrUnbounded:
        print("⚠ Ambiguous (INF_OR_UNBD). Retrying with DualReductions=0 …")
        solver.options['DualReductions'] = 0
        solver.reset()                    # clear the persistent state
        retry_result = solver.solve(biogas_model, tee=True)
        term = retry_result.solver.termination_condition
        print(f"→ New termination condition: {term}")

    # Now term is either INFEASIBLE, UNBOUNDED, or OPTIMAL/OTHER
    if term == TerminationCondition.infeasible:
        print("✘ Model is infeasible. Extracting IIS …")
        grb = solver._solver_model
        grb.computeIIS()
        grb.write("model_iis.ilp")
        print(" → IIS written to model.ilp.iis.")
    elif term == TerminationCondition.unbounded:
        print("⚠ MIP is unbounded (with integer vars).  → Relaxing integrality to extract a ray…")

        # --- 1) Rebuild the model (fresh copy) ---
        lp_model = build_model(cfg)

        # --- 2) Relax all integer (incl. binary) variables to continuous ---
        TransformationFactory('core.relax_integer_vars').apply_to(lp_model)

        # --- 3) Set solver options for “true” unbounded diagnosis ---
        lp_solver = SolverFactory('gurobi_persistent')
        lp_solver.set_instance(lp_model, symbolic_solver_labels=True)
        lp_solver.options['DualReductions'] = 0   # force a clean unbounded vs infeasible test
        lp_solver.options['InfUnbdInfo']   = 1   # request the ray

        # --- 4) Solve the continuous LP ---
        lp_result = lp_solver.solve(tee=True)
        lp_term   = lp_result.solver.termination_condition
        print(f"→ LP relaxation termination: {lp_term}")

        if lp_term == TerminationCondition.unbounded:
            grb_lp   = lp_solver._solver_model
            ray_coef = grb_lp.UnbdRay
            vars_lp  = grb_lp.getVars()

            # Invert Pyomo's internal map
            inv_map = {
                solver_var: pyomo_var
                for pyomo_var, solver_var in lp_solver._pyomo_var_to_solver_var_map.items()
            }

            print("\nNon-zero components of the unbounded ray (var : direction) and their Pyomo names:")
            for solver_var, coeff in zip(vars_lp, ray_coef):
                if abs(coeff) < 1e-8:
                    continue

                pyomo_var = inv_map.get(solver_var, None)
                print(f"  {solver_var.VarName:30s} : {coeff: .6e}"
                    f"   → Pyomo: {pyomo_var.name if pyomo_var is not None else '??'}")
        return
    elif term == TerminationCondition.optimal:
        print("✔ Model solved to optimality.")
    else:
        return (f"‼️ Unexpected termination condition: {term}")
    
    # Now you know you have a valid solution
    mip_obj = value(biogas_model.Obj)
    print(f"✔ MIP objective (total cost) = {mip_obj:,.2f}")

    # Assesment of variables
    ARTIFICIAL_UB = 1e9
    suspects = []

    for v in biogas_model.component_objects(Var, active=True):
        for idx in v:
            var = v[idx]
            if var.is_binary():
                continue
            else:
                if var.ub is not None:
                    val = value(var, exception=False)
                    if val is not None and math.isfinite(val):
                        if abs(val - var.ub) <= max(1e-3, 1e-6 * var.ub):
                            suspects.append((var.name, idx, val, var.ub))

    if suspects:
        print("Variables hitting upper bound:")
        for name, idx, val, ub in suspects:
            print(f"{name}{idx}: value={val}, ub={ub}")


    return biogas_model