# scripts/run_model.py
import argparse
from pyomo.environ import SolverFactory, Suffix, value, Var, Binary, TransformationFactory, Constraint
from pyomo.opt import TerminationCondition
from src.config           import ModelConfig
from src.model.builder    import build_model
from src.utils.export_resultT import export_results
from src.model.objective import debug_objective
from pyomo.repn import generate_standard_repn
from pyomo.core.base.constraint import Constraint
import csv
import time
from datetime import datetime, timedelta
from src.utils.max_contraint_violation import detect_max_constraint_violation
import pandas as pd
from src.utils.export_inputs import export_inputs
from dataclasses import asdict
from pathlib import Path
from src.utils.check_co2_wedge import check_co2_wedge, CO2WedgeConfig
from src.utils.trace_co2liq_price import trace_co2liq_price

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true', help="short horizon test run")
    p.add_argument('--n-test', type=int, help="hours to keep when --test is on")
    p.add_argument('--penalty', type=float, help="penalty multiplier for slack in objective")
    p.add_argument('--demand_target', type=lambda x: x.lower() == 'true', help="choose whether to enforce methanol annual demand target")
    p.add_argument('--sensitivity', type=lambda x: x.lower() == 'true', help="apply sensitivity case adjustments")
    p.add_argument('--green_electricity', type=lambda x: x.lower() == 'true', help="restrict grid electricity buy to <20 €/MWh")
    p.add_argument('--electricity_mandate', type=float, help="restricts electricity imports to a percent of consumption each hour")
    p.add_argument('--el_prod_to_grid', type=float, help="restricts electricity exports to a percent of generation each hour")
    p.add_argument('--multiple_scenarios', action='store_true', help="Run all Excel scenarios in 'scenarios' folder")
    p.add_argument('--data_file', type=str, help="Path to single Excel scenario file")

    return p.parse_args()

def run_model(cfg, scenario_name=None):
    start_time = time.time()
    print("==========================")
    print("Model Run Started")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================\n")

    print("Building Pyomo model ...\n")
    print("Config values:")
    for key, option in asdict(cfg).items():
        if key == "n_test" and not cfg.test_mode:
            continue
        print(f"{key}: {option}")

    model = build_model(cfg)
    model.name = 'GreenlabSkive_CK'
    print(f"Model {model.name} built successfully.\n")

    model.dual = Suffix(direction=Suffix.IMPORT)

    # 3) Solve the MIP
    solver = SolverFactory('gurobi_persistent')
    solver.set_instance(model, symbolic_solver_labels=True)
    solver.options['MIPGap'] = 0.05
    solver.options['Seed'] = 42
    solver.options['Threads'] = 1
    print("\nSolving MIP …\n")
    mip_result = solver.solve(model, tee=True)
    term = mip_result.solver.termination_condition
    print(f"\n→ Initial termination condition: {term}")
    print("\nMIP solve finished.\n")

    # Handle the ambiguous case and retry
    if term == TerminationCondition.infeasibleOrUnbounded:
        print("⚠ Ambiguous (INF_OR_UNBD). Retrying with DualReductions=0 …")
        solver.options['DualReductions'] = 0
        solver.reset()                    # clear the persistent state
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
    elif term == TerminationCondition.unbounded:
        print("⚠ MIP is unbounded (with integer vars).  → Relaxing integrality to extract a ray…")

        # --- 1) Rebuild the model (fresh copy) ---
        lp_model = build_model(cfg)
        lp_model.name = model.name + "_LPrelaxed"

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
        print("✔ Model solved to optimality.\n")
    else:
        return (f"‼️ Unexpected termination condition: {term}")
    
    # Now you know you have a valid solution
    mip_obj = value(model.Obj)
    print(f"✔ MIP objective (total cost) = {mip_obj:,.2f}")

    print("\nChecking constraint violations after MIP solve...")
    detect_max_constraint_violation(model, threshold=1e-4, top_n=10)

    # After solving the MIP, but before fixing binaries:
    for v in model.component_data_objects(Var, descend_into=True):
        if v.domain is Binary and v.value is not None:
            v.fix(v.value)

    print("\nRelaxing integer vars → pure LP …\n")
    TransformationFactory('core.relax_integer_vars').apply_to(model)

    # 5) Clear any old duals, then re‐solve as an LP to get duals
    print("Re‐solving as an LP to extract duals …\n")
    lp_solver = SolverFactory('gurobi')
    lp_result = lp_solver.solve(model, tee=False, suffixes=['dual'])
    lp_obj = value(model.Obj)
    print(f"→ LP objective (continuous, binaries fixed) = {lp_obj:,.2f}\n")
    print("LP solve finished.\n")

    check_co2_wedge(
        model,
        CO2WedgeConfig(
            area_co2="Skive",
            area_el="DK1",              # change to "Skive" if you truly price electricity at Skive
            tech_compressor="CO2Compressor",
            out_csv=f"results/CO2_wedge_check_{scenario_name}.csv",
        )
    )

    trace_co2liq_price(
        model,
        area="Skive",
        area_el="DK1",
        out_csv=f"results/trace_co2liq_price_{scenario_name}.csv",
    )

    # print("Hourly CO₂-balance breakdown for Skive and DK1:")
    # print(" Area | Time |   Buy   |  Inflow | Generation | Fueluse |  Sale  | Outflow |   LHS   |   RHS   | Imbalance | Dual ")
    # print("-----------------------------------------------------------------------------------------------------------")
    # for area in ['Skive','DK1']:
    #     for t in model.T:
    #         # 1) Buy/Sale
    #         buy_term  = value(model.Buy[area,'CO2',t])  if (area,'CO2') in model.buyE  else 0.0
    #         sale_term = value(model.Sale[area,'CO2',t]) if (area,'CO2') in model.saleE else 0.0

    #         # 2) Inflow / Outflow
    #         inflow  = sum(value(model.Flow[i, area, 'CO2', t])
    #                       for (i,j,e) in model.flowset if j==area and e=='CO2')
    #         outflow = sum(value(model.Flow[area, j, 'CO2', t])
    #                       for (i,j,e) in model.flowset if i==area and e=='CO2')

    #         # 3) Local gen / fuel use
    #         techs = [g for (a,g) in model.location if a==area]
    #         generation = sum(value(model.Generation[g,'CO2',t]) for g in techs if (g,'CO2') in model.f_out)
    #         fueluse    = sum(value(model.Fueluse[g,'CO2',t])    for g in techs if (g,'CO2') in model.f_in)

    #         # 4) Balance check
    #         lhs       = buy_term + inflow + generation
    #         rhs       = fueluse + sale_term + outflow
    #         imbalance = lhs - rhs

    #         # 5) Dual of the CO2-balance constraint
    #         con   = model.Balance[area, 'CO2', t]
    #         dualₚ = model.dual[con]

    #         # Print nicely
    #         print(f"{area:5} | {t:4} | {buy_term:7.2f} | {inflow:7.2f} | {generation:10.2f} |"
    #               f" {fueluse:7.2f} | {sale_term:6.2f} | {outflow:7.2f} |"
    #               f" {lhs:7.2f} | {rhs:7.2f} | {imbalance:9.2e} | {dualₚ:6.2f}")


    # # 5) Print duals for your CO2 balance constraint
    # #    (replace 'CO2_balance' and index set 'T' with whatever your builder uses)
    # print("CO₂‐balance duals (only for fuel = 'CO2'):")
    # for con in model.component_data_objects(Constraint, active=True):
    #     # Filter only the 'Balance' constraint family
    #     if con.parent_component().name == "Balance":
    #         a, e, t = con.index()        # unpack the (area, energy, time) tuple
    #         if e == "CO2":               # only for CO2
    #             π = model.dual[con]      # shadow price
    #             print(f"  area={a}, time={t}: dual = {π:,.4f}")

    # export_results_to_excel(model)
    results_path = f"results/Results_{scenario_name}.xlsx" if scenario_name else None
    inputs_path  = f"results/Inputs_{scenario_name}.xlsx"  if scenario_name else None
    print("Exporting to Excel ... ")
    export_results(model, cfg, path=results_path)
    export_inputs(model, cfg, path=inputs_path)
    # debug_objective(model, cfg)
    
    elapsed = time.time() - start_time
    print("\n==========================")
    print("Pyomo Model Run Completed")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    elapsed_td = timedelta(seconds=int(elapsed))
    print(f"Total runtime: {elapsed_td}")
    print("==========================")

    return model

def main():
    args = parse_args()
    defaults = ModelConfig()

    if args.multiple_scenarios:
        folder = Path("scenarios/")
        files = sorted(folder.glob("*.xlsx"))

        for file in files:
            print(f"\n=== Running scenario: {file.name} ===\n")
            cfg = ModelConfig(
                data_file=str(file),
                test_mode=args.test,
                n_test=args.n_test if args.n_test is not None else defaults.n_test,
                penalty=args.penalty if args.penalty is not None else defaults.penalty,
                demand_target=args.demand_target if args.demand_target is not None else defaults.demand_target,
                sensitivity=args.sensitivity if args.sensitivity is not None else defaults.sensitivity,
                green_electricity=args.green_electricity if args.green_electricity is not None else defaults.green_electricity,
                electricity_mandate=args.electricity_mandate if args.electricity_mandate is not None else defaults.electricity_mandate,
                el_prod_to_grid=args.el_prod_to_grid if args.el_prod_to_grid is not None else defaults.el_prod_to_grid,
            )
            run_model(cfg, scenario_name = file.stem.removeprefix("Data_"))

    else:
        cfg = ModelConfig(
            test_mode=args.test,
            n_test=args.n_test if args.n_test is not None else defaults.n_test,
            penalty=args.penalty if args.penalty is not None else defaults.penalty,
            demand_target=args.demand_target if args.demand_target is not None else defaults.demand_target,
            sensitivity=args.sensitivity if args.sensitivity is not None else defaults.sensitivity,
            green_electricity=args.green_electricity if args.green_electricity is not None else defaults.green_electricity,
            electricity_mandate=args.electricity_mandate if args.electricity_mandate is not None else defaults.electricity_mandate,
            el_prod_to_grid=args.el_prod_to_grid if args.el_prod_to_grid is not None else defaults.el_prod_to_grid,
            data_file=args.data_file or defaults.data_file,
        )

        scenario_path = Path(cfg.data_file)
        scenario_name = scenario_path.stem.removeprefix("Data_")

        run_model(cfg, scenario_name=scenario_name)

if __name__ == '__main__':
    model = main()
