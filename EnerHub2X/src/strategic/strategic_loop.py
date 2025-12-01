from pyomo.environ import value, SolverFactory, Suffix, Objective, maximize
from src.model.builder import build_model
from src.config import ModelConfig
import io
from contextlib import redirect_stdout

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
    print("[INFO] Building and solving initial centralized model...")
    base_model = build_model(cfg)
    base_model.dual = Suffix(direction=Suffix.IMPORT)
    solver = SolverFactory('gurobi')
    solver.solve(base_model, tee=False)
    print("\n[INFO] Centralized baseline solve completed.\n")

    # Extract centralized CO2 sales for comparison
    q_central = {}
    for t in base_model.T:
        for (area, fuel) in base_model.saleE:
            if fuel == co2_label:
                q_central[(area,t)] = value(base_model.Sale[area,fuel,t])
    print("[CHECK] Centralized CO2 sales:")
    if not q_central:
        print("[WARN] No CO2 sales found in centralized model.")
    for key in q_central:
        print(f"  {key}: {q_central[key]:.3f}")

    # Retrieve strategic suppliers
    strategic_suppliers = getattr(base_model, 'StrategicSuppliers', None)
    if not strategic_suppliers:
        raise RuntimeError("No StrategicSuppliers found on model. Check sets.py or loader configuration.")

    # Mapping tech -> area
    tech_to_area = {tech: area for (area, tech) in base_model.location}

    # Initialize current strategies (sales quantities)
    curr = _initialize_strategies(base_model, strategic_suppliers, tech_to_area, co2_label)
    print(f"\n[INFO] Initial strategies extracted for {len(strategic_suppliers)} suppliers.")

    # ------------------------------------------------------------
    # 2. BEST-RESPONSE ITERATION LOOP
    # ------------------------------------------------------------
    for iteration in range(1, max_iter+1):
        print(f"\n----- Iteration {iteration} -----")
        max_change = 0.0

        for tech in strategic_suppliers:
            # Build new submodel for supplier tech
            m = silent_build_model(cfg)
            m.dual = Suffix(direction=Suffix.IMPORT)

            # Fix competitors' sales
            _fix_competitor_sales(m, tech, strategic_suppliers, tech_to_area, curr, co2_label)

            # TODO: Modify objective to firm profit_i
            # profit_i = sum_t [ price(t)*Sale_i(t) - cost_i_gen(t) ].
            # Easiest: set price param to inverse demand p(t) = a - b*(sum_all_sales)
            # But since competitor sales are fixed we can compute demand price as function of this supplier's sale variable.
            # For simplicity, we just keep the original objective but let solver choose the best Sale for tech by not fixing its sale variables,
            # and ensure no other decision variables allow arbitrage. This is approximate but often works for simple cases.
            # Solve BR
            # _define_strategic_objective(m, tech, strategic_suppliers, tech_to_area, curr, co2_label)

            # Solve the submodel profit maximization for this suplier i
            solver.solve(m, tee=False)

            # Extract and update this supplier's best response
            change = _update_strategy(m, tech, tech_to_area, curr, co2_label, damping)
            max_change = max(max_change, change)

        print(f"[ITER {iteration}] Max change = {max_change:.6f}")

        # Convergence test
        if max_change < tol:
            print(f"\n[INFO] Converged after {iteration} iterations (tol={tol}).\n")
            print(curr)
            break
    else:
        print("[WARN] Max iterations reached without full convergence.\n")

    # ------------------------------------------------------------
    # 3. FINALIZATION PHASE
    # ------------------------------------------------------------
    print("[INFO] Building final model with fixed strategic sales...")
    final_model = silent_build_model(cfg)
    final_model.dual = Suffix(direction=Suffix.IMPORT)
    _fix_all_sales(final_model, strategic_suppliers, tech_to_area, curr, co2_label)

    print("\n[INFO] Solving final full model (MIP)...\n")
    final_solver = SolverFactory('gurobi_persistent')
    final_solver.set_instance(final_model, symbolic_solver_labels=True)
    final_solver.options['MIPGap'] = 0.05
    final_solver.solve(final_model, tee=True)

    print("\n================= Cournot Loop Completed =================\n")
    print(f"Converged after {iteration} iterations (tol={tol}).")

    print("\n[CHECK] Comparing centralized vs strategic results...")

    # Central model
    base_model = silent_build_model(cfg)
    solver.solve(base_model, tee=False)
    q_central = {}
    for (area, fuel) in base_model.saleE:
        if fuel == co2_label:
            q_central[(area,t)] = value(base_model.Sale[area,fuel,t])

    # Strategic model
    q_strat = {}
    for (area, fuel) in final_model.saleE:
        if fuel == co2_label:
            q_strat[(area,t)] = value(final_model.Sale[area,fuel,t])

    if not q_central:
        print("[WARN] No CO2 sales found in centralized model for comparison.")
    for key in q_central:
        print(f"  {key}: central={q_central[key]:.3f}, strategic={q_strat[key]:.3f}, Δ={q_strat[key]-q_central[key]:.3e}")
    
    return final_model, curr


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