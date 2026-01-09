# src/strategic/strategic_loop.py

from pyomo.environ import value, SolverFactory, Suffix, Objective, maximize
from src.model.builder import build_model
from src.config import ModelConfig
import io
from contextlib import redirect_stdout
from src.strategic.submodel_biogas import build_biogas_model
from src.strategic.submodel_methanol import build_methanol_model  # Assuming you implement this

def run_cournot(cfg: ModelConfig, tol=1e-3, max_iter=30, damping=0.6, co2_label='CO2Comp'):
    """
    Iterative Cournot best-response algorithm for CO2 market.
    Each strategic supplier adjusts its sale quantity given competitors' fixed quantities.
    Convergence is declared when all strategy changes fall below `tol`.
    """
    print("\n================= Cournot Strategic Loop =================\n")

    # ------------------------------------------------------------
    # 1. INITIALIZATION PHASE
    # ------------------------------------------------------------
    print("[INFO] Building and solving initial centralized model...")
    base_model = silent_build_model(cfg)  # Use silent_build_model to avoid prints
    base_model.dual = Suffix(direction=Suffix.IMPORT)
    solver = SolverFactory('gurobi')
    solver.solve(base_model, tee=False)
    print("[INFO] Centralized baseline solve completed.")

    # Retrieve strategic actors (assume list of techs, e.g., ['BiogasUpgrade'])
    # strategic_suppliers = getattr(base_model, 'StrategicSuppliers', ['BiogasUpgrade'])
    # strategic_demanders = getattr(base_model, 'StrategicDemanders', ['MethanolSynthesis'])  
    
    # As of now, carrier mix is defined such that supply of CO2Comp is = {CO2Compressor, CO2Storage} and demand is = {MethanolSynthesis}
    strategic_suppliers = ['CO2Compressor', 'CO2Storage']
    strategic_demanders = ['MethanolSynthesis']
  
    # Extract CO2 comp use (demand) and duals (willingness to pay) for curve construction
    co2_comp_use = {t: value(base_model.Fueluse['MethanolSynthesis', co2_label, t]) for t in base_model.T if ('MethanolSynthesis', co2_label, t) in base_model.Fueluse}
    co2_duals = {t: base_model.dual.get(base_model.Balance['Skive', 'CO2', t], 0.0) if ('Skive', 'CO2', t) in base_model.Balance.index_set() else 0.0 for t in base_model.T} 

    # Mapping tech -> area
    tech_to_area = {tech: area for (area, tech) in base_model.location}
    # area = tech_to_area.get(tech)

    # Initialize supply strategies using CO2Comp generation (supply)
    # curr = _initialize_strategies(base_model, strategic_suppliers, tech_to_area, co2_label)

    curr = {} # supply current strategy
    for tech in strategic_suppliers:
        curr[tech] = {}
        for t in base_model.T:
            # Use Generation for CO2-producing techs as proxy for sales
            gen_key = (tech, co2_label, t) if (tech, co2_label) in base_model.f_out else None
            curr[tech][t] = value(base_model.Generation[gen_key]) if gen_key in base_model.Generation else 0.0

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
        capacity = co2_comp_use.get(t, 0.0)
        price = co2_duals[t] 
        block3 = {"block": 3, "price": price, "capacity": capacity}
        demand_price_blocks[t] = sorted(dummy_blocks + [block3], key=lambda x: -x['price'])
        for i, block in enumerate(demand_price_blocks[t]):
            block['block'] = i + 1  # Re-index blocks

    print(f"\n[INFO] Strategies and demand curves intialized.")

    # ------------------------------------------------------------ 
    # 2. BEST-RESPONSE ITERATION LOOP
    # ------------------------------------------------------------
    for iteration in range(1, max_iter+1):
        print(f"\n----- Iteration {iteration} -----")
        max_change = 0.0

        # Optimize supply-side submodel (biogas)
        # Assuming only one strategic supplier for simplicity; extend as needed
        biogas_submodel = build_biogas_model(cfg, demand_price_blocks)
        biogas_submodel.dual = Suffix(direction=Suffix.IMPORT)
        solver.solve(biogas_submodel, tee=False)

        # _fix_competitor_sales(biogas_submodel, tech, strategic_suppliers, tech_to_area, curr, co2_label)

        # Extract and update the supplier's best response
        for t in biogas_submodel.T:

            # co2_sell = value(biogas_submodel.CO2_TotalSell[t]) if t in biogas_submodel.CO2_TotalSell else 0.0
            # co2_price = value(biogas_submodel.CO2_MarketPrice[t]) if t in biogas_submodel.CO2_MarketPrice else 0.0

            co2_sell = 0.0
            for tech in strategic_suppliers:
                # Use Generation for CO2-producing techs as proxy for sales
                gen_key = (tech, co2_label, t) if (tech, co2_label) in biogas_submodel.f_out else None
                new_val = value(biogas_submodel.Generation[gen_key]) if gen_key in biogas_submodel.Generation else 0.0
                old_val = curr[tech].get(t, 0.0)

                updated = damping * new_val + (1 - damping) * old_val
                curr[tech][t] = updated

                max_change = max(max_change, abs(new_val - old_val))

                co2_sell += new_val

            if co2_sell != value(biogas_submodel.CO2_TotalSell[t]):
                print(f"[WARN] Mismatch in total CO2 sell at time {t}")

        # Check convergence
        print(f"[ITER {iteration}] Max change = {max_change:.6f}")
        if max_change < tol:
            print(f"\n[INFO] Converged after {iteration} iterations (tol={tol}).\n")
            print("\n================= Cournot Loop Completed =================\n")
            break

        # Optimize demand-side submodel (methanol) for next iteration
        methanol_submodel = build_methanol_model(cfg, curr, dummy_blocks)
        methanol_submodel.dual = Suffix(direction=Suffix.IMPORT)
        solver.solve(methanol_submodel, tee=False)

        demand_price_blocks = {}
        # Update demand_price_blocks for next iteration based on new CO2 demand
        for t in methanol_submodel.T:
            capacity = value(methanol_submodel.Fueluse['MethanolSynthesis', co2_label, t]) if ('MethanolSynthesis', co2_label, t) in methanol_submodel.Fueluse else 0.0
            price = methanol_submodel.dual.get(methanol_submodel.Balance['Skive', 'CO2', t], 0.0) if ('Skive', 'CO2', t) in methanol_submodel.Balance.index_set() else 0.0
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
    # print("[INFO] Building final model with fixed strategic sales...")
    # final_model = silent_build_model(cfg)
    # final_model.dual = Suffix(direction=Suffix.IMPORT)

    # # Fix all strategic suppliers' sales
    # _fix_all_sales(final_model, strategic_suppliers, tech_to_area, curr, co2_label)

    # print("\n[INFO] Solving final full model (MIP)...\n")
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
    return model