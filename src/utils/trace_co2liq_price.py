# src/utils/trace_co2liq_price.py
from __future__ import annotations
import pandas as pd
from pyomo.environ import value


def _dual(model, con):
    return float(model.dual.get(con, float("nan")))


def trace_co2liq_price(
    model,
    area="Skive",
    e_liq="CO2_Liq",
    e_el="Electricity",
    tech_storage="CO2Storage",
    tech_compressor="CO2Compressor",
    area_el="DK1",
    eps=1e-8,
    out_csv="results/trace_co2liq_price.csv",
    n_rows=200,
):
    """
    Builds an hour-by-hour diagnostic table showing:
      - CO2_Liq balance dual => p_liq = -dual
      - compressor activity and implied wedge term (0.07 * p_el)
      - storage activity (charge/discharge/volume) and key storage duals:
          lambda_t = dual(ProductionStorage[tech_storage,t])
          mu_t     = dual(VolumeUpper[tech_storage,t])
          nu       = dual(TerminalSOC[tech_storage])
    """

    rows = []
    nu = _dual(model, model.TerminalSOC[tech_storage]) if tech_storage in model.G_s else float("nan")

    for t in model.T:
        # --- CO2_Liq nodal dual/price ---
        if (area, e_liq, t) in model.Balance:
            d_liq = _dual(model, model.Balance[area, e_liq, t])
        else:
            d_liq = float("nan")
        p_liq = -d_liq if pd.notna(d_liq) else float("nan")

        # --- Electricity nodal dual/price (for compressor wedge) ---
        if (area_el, e_el, t) in model.Balance:
            d_el = _dual(model, model.Balance[area_el, e_el, t])
        else:
            d_el = float("nan")
        p_el = -d_el if pd.notna(d_el) else float("nan")

        # --- Compressor activity ---
        comp_out = float(value(model.Generation[tech_compressor, "CO2_Comp", t])) if (tech_compressor, "CO2_Comp") in model.f_out else 0.0
        comp_in_liq = float(value(model.Fueluse[tech_compressor, e_liq, t])) if (tech_compressor, e_liq) in model.f_in else 0.0
        comp_in_el  = float(value(model.Fueluse[tech_compressor, e_el, t])) if (tech_compressor, e_el) in model.f_in else 0.0
        wedge_el = 0.07 * p_el if pd.notna(p_el) else float("nan")  # your carrier-mix coefficient

        # --- Storage activity & duals ---
        if tech_storage in model.G_s:
            vol = float(value(model.Volume[tech_storage, t]))
            chg = float(value(model.Fuelusetotal[tech_storage, t]))  # charging amount (pre-eff)
            # discharge as defined in storage_balance_rule:
            discharge = 0.0
            for (gg, e) in model.f_out:
                if gg == tech_storage:
                    discharge += float(value(model.Generation[tech_storage, e, t])) * float(value(model.out_frac[tech_storage, e]))
            # Storage balance dual lambda_t:
            lam = _dual(model, model.ProductionStorage[tech_storage, t])
            # SOC max dual mu_t:
            mu = _dual(model, model.VolumeUpper[tech_storage, t])
        else:
            vol = chg = discharge = lam = mu = float("nan")

        # Regime flags
        comp_on = comp_in_liq > eps or comp_out > eps
        store_charge_on = chg > eps
        store_discharge_on = discharge > eps
        any_activity = comp_on or store_charge_on or store_discharge_on

        rows.append({
            "t": str(t),
            "dual_CO2_Liq": d_liq,
            "p_CO2_Liq": p_liq,
            "dual_Electricity": d_el,
            "p_Electricity": p_el,
            "comp_in_CO2_Liq": comp_in_liq,
            "comp_in_Electricity": comp_in_el,
            "comp_out_CO2_Comp": comp_out,
            "wedge_0.07_p_el": wedge_el,
            "store_volume": vol,
            "store_charge_Fuelusetotal": chg,
            "store_discharge": discharge,
            "dual_storage_balance_lambda": lam,
            "dual_soc_max_mu": mu,
            "dual_terminal_soc_nu": nu,
            "comp_on": comp_on,
            "store_charge_on": store_charge_on,
            "store_discharge_on": store_discharge_on,
            "any_activity": any_activity,
        })

    df = pd.DataFrame(rows)

    # Helpful: highlight “idle” hours that still have nonzero p_liq
    df["idle_but_priced"] = (~df["any_activity"]) & (df["p_CO2_Liq"].abs() > 1e-6)

    # Save & print a quick summary
    df.to_csv(out_csv, index=False)
    print(f"→ wrote {out_csv}")
    print("Counts:")
    print(df[["comp_on","store_charge_on","store_discharge_on","any_activity","idle_but_priced"]].sum())

    # show first n rows with idle-but-priced behavior
    suspicious = df[df["idle_but_priced"]].head(n_rows)
    if not suspicious.empty:
        print("\nFirst idle-but-priced hours:")
        print(suspicious[["t","p_CO2_Liq","dual_CO2_Liq","dual_storage_balance_lambda","dual_soc_max_mu","dual_terminal_soc_nu"]])

    return df