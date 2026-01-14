# src/utils/export_results.py

import pandas as pd
from pyomo.environ import value
from pathlib import Path
from src.config import ModelConfig
from collections import defaultdict
import re

def export_results(model, cfg: ModelConfig, path: str = None):
    """
    Export GAMS‐style ResultT, ResultF and ResultA tables to Excel,
    plus “ResultTsum”, “ResultFsum” and “ResultAsum” pivoted summaries.

    Sheets produced:
      - ResultT_all    (hourly: Operation, Volume, Costs_EUR, Startcost_EUR, Variable_OM_cost_EUR)
      - ResultTsum     (pivot: one row per (Result, tech), columns = each energy, values = sum over hours;
                        “Result” in the same block‐order as the hourly sheet)
      - Flows          (hourly: areaFrom, areaTo, energy)
      - ResultFsum     (pivot: one row per (areaFrom, areaTo), columns = each energy, values = sum over hours)
      - ResultA_all    (hourly: Buy, Sale, Demand, Import_price_EUR, Export_price_EUR, Buy_EUR, Sale_EUR)
      - ResultAsum     (pivot: one row per (Result, area), columns = each energy,
                        values = sum over hours except for price rows:
                        – for energy=="Electricity", take the average over all hours
                        – for all other energies, take the first‐hour price)
      - ResultC        (hourly capacity factors)
      - ResultCsum     (Average CF and FLH)
    """
    # 1) compute base path
    if path is None:
        project_root = Path(__file__).parents[2]
        default = project_root / "results" / "Results.xlsx"
    else:
        default = Path(path)
    default.parent.mkdir(parents=True, exist_ok=True)

    base, suffix, folder = default.stem, default.suffix, default.parent
    if cfg.test_mode:
        base = "test_" + base

    # 2) Common time index
    times = list(model.T)
    time_cols = [str(t) for t in times]
    ntimes = len(times)

    # --- build ResultT blocks ---
    pairs = set(model.f_in) | set(model.f_out)

    # 3a) Operation
    op = []
    for g, e in pairs:
        row = {'Result': 'Operation', 'tech': g, 'energy': e}
        for t in times:
            gen = value(model.Generation[g, e, t]) if (g, e) in model.f_out else 0
            use = value(model.Fueluse[g, e, t])      if (g, e) in model.f_in  else 0
            row[str(t)] = gen - use
        op.append(row)
    df_op = pd.DataFrame(op)

    # 3b) Volume
    vol = []
    for g in model.G_s:
        for e in (f for (gg, f) in model.f_out if gg == g):
            row = {'Result': 'Volume', 'tech': g, 'energy': e}
            for t in times:
                row[str(t)] = value(model.Volume[g, t])
            vol.append(row)
    df_vol = pd.DataFrame(vol)

    # 3c) Costs_EUR
    cost = []
    for g, e in pairs:
        row = {'Result': 'Costs_EUR', 'tech': g, 'energy': e}
        for t in times:
            imp_qty  = value(model.Fueluse[g, e, t])    if (g, e) in model.f_in  else 0
            sale_qty = value(model.Generation[g, e, t]) if (g, e) in model.f_out else 0
            imp_price  = sum(model.price_buy[a, e, t]  for a in model.A if (a, e) in model.buyE)
            sale_price = sum(model.price_sale[a, e, t] for a in model.A if (a, e) in model.saleE)
            row[str(t)] = imp_qty * imp_price - sale_qty * sale_price
        cost.append(row)
    df_cost = pd.DataFrame(cost)
    # print('\nATTENTION:')
    # print('df_cost for things you dont import or export is wrong (e.g. cost from on-site RES)\n'
    #       'The model prints out as cost, the ELECTRICITY produced by RES * export_price of electricity on the market.\n')

    # 3d) Startcost_EUR
    start = []
    for g in model.G:
        row = {'Result': 'Startcost_EUR', 'tech': g, 'energy': 'system_cost'}
        for t in times:
            row[str(t)] = value(model.Startcost[g, t])
        start.append(row)
    df_start = pd.DataFrame(start)

    # 3e) Variable_OM_cost_EUR
    varom = []
    for g in model.G:
        row = {'Result': 'Variable_OM_cost_EUR', 'tech': g, 'energy': 'system_cost'}
        # Find all exported energies for this technology
        export_fuels = [e for (gg, e) in model.TechToEnergy if gg == g]
        for t in times:
            gen_sum = sum(value(model.Generation[g, e, t]) for e in export_fuels)
            row[str(t)] = gen_sum * model.cvar[g]
        varom.append(row)
    df_varom = pd.DataFrame(varom)

    # concatenate all ResultT (hourly)
    df_T = pd.concat([df_op, df_vol, df_cost, df_start, df_varom], ignore_index=True)
    # --- enforce block‐order on Result *and* tech before sorting ---
    # 1) Result-block order exactly as in your GAMS Tech.inc export
    block_order_T = [
        'Operation',
        'Volume',
        'Costs_EUR',
        'Startcost_EUR',
        'Variable_OM_cost_EUR'
    ]
    df_T['Result'] = pd.Categorical(
        df_T['Result'],
        categories=block_order_T,
        ordered=True
    )

    # 2) Tech order from model.G (already ordered=True in your sets)
    tech_order = list(model.G)
    df_T['tech'] = pd.Categorical(
        df_T['tech'],
        categories=tech_order,
        ordered=True
    )

    # 3) Now sort once by the two categoricals + energy
    df_T.sort_values(['Result','tech','energy'], inplace=True)

    # 4) Re‐slice columns so time‐cols remain at the end
    df_T = df_T[['Result','tech','energy'] + time_cols]

    # --- build Flows sheet (hourly) ---
    flows = []
    for ao, ai, f in model.flowset:
        row = {'areaFrom': ao, 'areaTo': ai, 'energy': f}
        for t in times:
            row[str(t)] = value(model.Flow[ao, ai, f, t])
        flows.append(row)
    df_F = pd.DataFrame(flows)
    df_F.sort_values(['areaFrom','areaTo','energy'], inplace=True)
    df_F = df_F[['areaFrom','areaTo','energy'] + time_cols]

    # ------------------------------------------------
    # --- RESULTTsum: pivot AND enforce block order ---
    # ------------------------------------------------
    df_Tsum = (
        df_T
          .set_index(['Result','tech','energy'])[time_cols]
          .sum(axis=1)             # sum each (Result,tech,energy) over all t
          .unstack(fill_value=0)   # pivot so “energy” becomes columns
          .reset_index()           # bring “Result” & “tech” back as columns
    )

    # Enforce the exact block‐order for “Result” in df_Tsum
    block_order_T = [
        'Operation',
        'Volume',
        'Costs_EUR',
        'Startcost_EUR',
        'Variable_OM_cost_EUR'
    ]
    df_Tsum['Result'] = pd.Categorical(
        df_Tsum['Result'],
        categories=block_order_T,
        ordered=True
    )

    # 1) enforce Tech.inc order on the “tech” column
    tech_order = list(model.G)
    df_Tsum['tech'] = pd.Categorical(
        df_Tsum['tech'],
        categories=tech_order,
        ordered=True
    )
    # 2) now sort by Result (block), then by that tech order
    df_Tsum.sort_values(['Result','tech'], inplace=True)


    # --- build ResultA sheet (hourly) ---
    A_rows = []

    # 1) Buy & 2) Sale quantities
    for res, varset in (('Buy',  model.buyE), ('Sale', model.saleE)):
        for a, e in varset:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                row[str(t)] = (
                    value(model.Buy[a, e, t]) if res == 'Buy'
                    else value(model.Sale[a, e, t])
                )
            A_rows.append(row)

    # 3) Demand – only truly initialized & non‐zero
    raw_demand = dict(model.demand.items())
    dem_pairs = sorted({
        (a, e)
        for (a, e, t), val in raw_demand.items()
        if val != 0
    })
    for a, e in dem_pairs:
        row = {'Result': 'Demand', 'area': a, 'energy': e}
        for t in times:
            row[str(t)] = raw_demand.get((a, e, t), 0)
        A_rows.append(row)

    # 4) Import_price_EUR & 5) Export_price_EUR
    for res, price_param, sel in (
        ('Import_price_EUR',  model.price_buy,  model.buyE),
        ('Export_price_EUR',  model.price_sale, model.saleE)
    ):
        for a, e in sel:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                row[str(t)] = price_param[a, e, t]
            A_rows.append(row)

    # 6) Buy_EUR & 7) Sale_EUR
    for res, varset, price_param in (
        ('Buy_EUR',  model.buyE,  model.price_buy),
        ('Sale_EUR', model.saleE, model.price_sale)
    ):
        for a, e in varset:
            row = {'Result': res, 'area': a, 'energy': e}
            for t in times:
                qty   = (
                    value(model.Buy[a, e, t])
                    if res == 'Buy_EUR'
                    else value(model.Sale[a, e, t])
                )
                price = price_param[a, e, t]
                row[str(t)] = qty * price
            A_rows.append(row)

    df_A = pd.DataFrame(A_rows)

    # enforce the exact block‐order for df_A
    block_order_A = [
        'Buy',
        'Sale',
        'Demand',
        'Import_price_EUR',
        'Export_price_EUR',
        'Buy_EUR',
        'Sale_EUR'
    ]
    df_A['Result'] = pd.Categorical(
        df_A['Result'],
        categories=block_order_A,
        ordered=True
    )
    df_A.sort_values(['Result','area','energy'], inplace=True)
    df_A = df_A[['Result','area','energy'] + time_cols]


    # ------------------------------------------------------------
    # --- RESULTAsum: pivot but adjust “Electricity” price rows ---
    # ------------------------------------------------------------
    price_mask = df_A['Result'].isin(['Import_price_EUR','Export_price_EUR'])

    # (a) Build a small DataFrame that holds exactly one price‐value per
    #     (Result, area, energy):
    price_rows = df_A[price_mask].copy()

    # Compute “PriceValue” as:
    #  - If energy == "Electricity", average over all hours
    #  - Else, take the first hour’s price (time_cols[0])
    price_rows['PriceValue'] = price_rows.apply(
        lambda row: row[time_cols].mean()
                    if row['energy'] == 'Electricity'
                    else row[time_cols[0]],
        axis=1
    )

    # Pivot this “PriceValue” table so that each “energy” becomes its own column:
    df_price = (
        price_rows
          .set_index(['Result','area','energy'])['PriceValue']
          .unstack(level='energy', fill_value=0)
          .reset_index()
    )
    # Now df_price has columns:
    #    [ 'Result', 'area', '<energy1>', '<energy2>', … ]
    # where for (Import_price_EUR, "DK1", "Electricity"), the cell is
    # the *average* of all hourly Import_price_EUR["DK1","Electricity",t].

    # (b) Build the non‐price sums exactly as before:
    df_nonprice = (
        df_A[~price_mask]
          .set_index(['Result','area','energy'])[time_cols]
          .sum(axis=1)             # sum each (Result,area,energy) over all t
          .unstack(fill_value=0)   # pivot so “energy” becomes columns
          .reset_index()           # bring “Result” & “area” back as columns
    )

    # (c) Combine them and re‐apply block order
    df_Asum = pd.concat([df_nonprice, df_price], ignore_index=True)
    df_Asum['Result'] = pd.Categorical(
        df_Asum['Result'],
        categories=block_order_A,
        ordered=True
    )
    df_Asum.sort_values(['Result','area'], inplace=True)

    # --- Duals sheet: 1) hourly CO2, 2) weekly methanol duals ---
    # 1) Hourly CO2 duals
    co2_rows = []
    for (area, energy, t) in model.Balance.index_set():
        if energy == 'CO2_Comp' and area == 'Skive':
            con       = model.Balance[area, energy, t]
            dual_val  = model.dual.get(con, 0.0)
            co2_rows.append({
                'Area':   area,
                'Energy': energy,
                'Time':   t,
                'Dual':   dual_val
            })
    df_co2 = pd.DataFrame(co2_rows, columns=['Area','Energy','Time','Dual'])

    # for t in times:
    #     row = {'Time': str(t)}
    #     for area in ['Skive']:
    #         idx = (area, 'CO2', t)
    #         if idx in model.Balance.index_set():
    #             dual_val = model.dual.get(model.Balance[idx], 0.0)
    #         else:
    #             dual_val = 0.0
    #         row[f"CO2_{area}"] = dual_val
    #     co2_rows.append(row)
    # df_co2 = pd.DataFrame(co2_rows)


    # === Export duals for all enforced fuel demands ===
    if hasattr(model, "TargetDemand") and hasattr(model, "DemandFuel"):
        def sort_key(item):
            step, _ = item
            match = re.search(r'\d+', step)
            return int(match.group()) if match else float('inf')

        dual_rows = []
        for (step, af) in sorted(model.DemandFuel, key=sort_key):
            constraint = model.TargetDemand[step, af]
            dual_val = model.dual.get(constraint, float("nan"))  # NaN if no dual
            dual_rows.append({
                "Step": step,
                "Area.Fuel": af,
                "Dual Value": dual_val
            })

        df_duals = pd.DataFrame(dual_rows)

    # ----------------------------------------------------------------
    # --- ResultC (capacity factors) ---------------------------------
    # ----------------------------------------------------------------
    C_rows = []
    for tech,fuel in model.TechToEnergy:
        cap = value(model.original_capacity[tech])
        row = {'Result': 'CapacityFactor', 'tech': tech}
        for t in times:
            gen= value(model.Generation[tech, fuel, t])
            row[str(t)] = gen / cap if cap != 0 else 0
        C_rows.append(row)

    df_C_hourly = pd.DataFrame(C_rows, columns=['Result','tech'] + time_cols)

    summary_cf = []
    summary_flh = []
    for tech,fuel in model.TechToEnergy:
        cap = value(model.original_capacity[tech])
        total_gen = sum(
            value(model.Generation[tech, fuel, t]) 
            for t in times
        )
        avg_cf = total_gen / (cap * ntimes) if cap != 0 else 0
        summary_cf.append({'tech': tech, 'Average_CF': avg_cf})
        summary_flh.append({'tech': tech, 'FLH': avg_cf * ntimes})

    avg_cf_dict = { r['tech']: r['Average_CF'] for r in summary_cf }
    flh_dict    = { r['tech']: r['FLH']        for r in summary_flh }

    # Make the 2×N DataFrame
    df_Csum = pd.DataFrame(
        [ avg_cf_dict, flh_dict ],
        index=['CapacityFactor','FLH']
    )
    df_Csum.index.name = 'Result'

    # 9) Objective decomposition (total over all time‐steps, by element)
    decomp = []

    #  a) Fuel imports (“Buy_…”) are costs → negative contributions
    for (a, e) in model.buyE:
        tot = sum(
            value(model.price_buy[a, e, t] * model.Buy[a, e, t])
            for t in times
        )
        decomp.append({
            "Element": f"Buy_{e}",
            "Contribution": - tot
        })

    #  b) Fuel sales (“Sell_…”) are revenues → positive
    for (a, e) in model.saleE:
        tot = sum(
            value(model.price_sale[a, e, t] * model.Sale[a, e, t])
            for t in times
        )
        decomp.append({
            "Element": f"Sell_{e}",
            "Contribution": tot
        })

    # #  c) Variable O&M on tech→energy
    # tot_varom = sum(
    #     value(model.Generation[g, e, t] * model.cvar[g])
    #     for (g, e) in model.TechToEnergy
    #     for t in times
    # )
    # decomp.append({"Element": "Variable_OM", "Contribution": - tot_varom})

    # c) Variable O&M per technology
    varom_by_tech = defaultdict(float)
    for (g, e) in model.TechToEnergy:
        for t in times:
            varom_by_tech[g] += value(model.Generation[g, e, t]) * model.cvar[g]

    # Append each tech's contribution to decomposition
    for g, val in varom_by_tech.items():
        decomp.append({
            "Element": f"Variable_OM_{g}",
            "Contribution": -val
        })

    #  d) Startup costs
    tot_start = sum(
        value(model.Startcost[g, t])
        for g in model.G
        for t in times
    )
    decomp.append({"Element": "Startup", "Contribution": - tot_start})

    # e) Slack penalties (skip any un‐initialized vars)
    tot_slack_imp = 0.0
    for (a, e, t) in model.DemandSet:
        var = model.SlackDemandImport[a, e, t]
        if var.value is not None:
            tot_slack_imp += value(var)

    tot_slack_exp = 0.0
    for (a, e, t) in model.DemandSet:
        var = model.SlackDemandExport[a, e, t]
        if var.value is not None:
            tot_slack_exp += value(var)

    penalty = cfg.penalty


    decomp.append({
        "Element": "Slack",
        "Contribution": tot_slack_imp + tot_slack_exp
    })

    decomp.append({
        "Element": "Slack Cost",
        "Contribution": - penalty * (tot_slack_imp + tot_slack_exp)
    })

    # Aggregate slack and cost for all demand-driven fuels
    fuel_slack_totals = defaultdict(float)

    for (step, af) in model.DemandFuel:
        var = model.SlackTarget[step, af]
        if var.value is not None:
            fuel_slack_totals[af] += value(var)

    for af, slack_val in fuel_slack_totals.items():
        decomp.append({
            "Element": f"Slack {af}",
            "Contribution": slack_val
        })
        decomp.append({
            "Element": f"Slack {af} Cost",
            "Contribution": - penalty * slack_val
        })

    # Add TotalCost as the sum of all contributions
    total_profit = sum(entry["Contribution"] for entry in decomp)
    decomp.append({
        "Element": "TotalProfit",
        "Contribution": total_profit
    })

    df_decomp = pd.DataFrame(decomp)

    # ----------------------------------------------------------------
    # --- Write all sheets (including updated “sum” sheets) ---------
    # ----------------------------------------------------------------
    i = 0
    while True:
        filename = f"{base}{'' if i == 0 else f'({i})'}{suffix}"
        output = folder / filename
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 1) hourly ResultT
                df_T.to_excel(writer, sheet_name='ResultT', index=False)
                # 2) summed ResultT
                df_Tsum.to_excel(writer, sheet_name='ResultTsum', index=False)
                # 3) hourly Flows
                df_F.to_excel(writer, sheet_name='ResultF', index=False)
                # 4) summed Flows
                df_Fsum = (
                    df_F
                    .set_index(['areaFrom','areaTo','energy'])[time_cols]
                    .sum(axis=1)             # sum each (areaFrom, areaTo, energy) over all t
                    .unstack(fill_value=0)   # pivot so “energy” becomes columns
                    .reset_index()           # bring “areaFrom” & “areaTo” back as columns
                )
                df_Fsum.to_excel(writer, sheet_name='ResultFsum', index=False)
                # 5) hourly ResultA
                df_A.to_excel(writer, sheet_name='ResultA', index=False)
                # 6) summed ResultA
                df_Asum.to_excel(writer, sheet_name='ResultAsum', index=False)
                # 7) hourly capacity factors
                df_C_hourly.to_excel(
                    writer,
                    sheet_name='ResultC',
                    index=False
                )
                # 8) summary capacity factors
                df_Csum.to_excel(
                    writer,
                    sheet_name='ResultCsum'
                )
                # 9) Duals – hourly and weekly dual values
                # write hourly CO2 at the top
                df_co2.to_excel(writer, sheet_name='Duals', index=False, startrow=0, startcol=0)

                df_duals.to_excel(writer, sheet_name='Duals', index=False, startrow=0, startcol=5)
                    

                # 10) Objective function decomposition
                df_decomp.to_excel(writer, sheet_name="ObjDecomp", index=False)

            break

        except PermissionError:
            i += 1
            if i > 100:
                raise RuntimeError("Could not write after 100 attempts")
            print(f"⚠️  {output.name} is in use—trying {base}({i}){suffix}")

    
    print("Results exported successfully.")
    print(f"File: {output.resolve()}")


