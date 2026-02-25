# src/utils/check_co2_wedge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd
from pyomo.environ import value


@dataclass
class CO2WedgeConfig:
    area_co2: str = "Skive"        # CO2_Liq and CO2_Comp node (your boundary area)
    area_el: str = "DK1"           # where electricity is priced (often DK1 in your model)
    tech_compressor: str = "CO2Compressor"
    e_liq: str = "CO2_Liq"
    e_comp: str = "CO2_Comp"
    e_el: str = "Electricity"
    run_threshold: float = 1e-6    # compressor considered "on" if output > threshold
    out_csv: Optional[str] = None  # e.g. "results/CO2_wedge_check.csv"


def _dual(model, a: str, e: str, t) -> float:
    """Safe dual lookup from the Balance constraint."""
    idx = (a, e, t)
    if idx not in model.Balance:
        return float("nan")
    return float(model.dual.get(model.Balance[idx], float("nan")))


def _price_from_dual(dual_val: float) -> float:
    """
    Your objective is maximize, and Balance is written as LHS==RHS (supply==use).
    In that convention, an economic 'nodal price' is typically p = -dual.
    """
    if pd.isna(dual_val):
        return float("nan")
    return -dual_val


def check_co2_wedge(model, cfg: CO2WedgeConfig) -> pd.DataFrame:
    """
    Checks whether the nodal price wedge satisfies:

        p(CO2_Comp) ≈ alpha_liq * p(CO2_Liq) + alpha_el * p(Electricity)

    when the compressor is running, where alpha_* are inferred from in_frac and Fe
    as "inputs per 1 unit of CO2_Comp output".

    Returns a DataFrame with hour-by-hour diagnostics.
    """

    g = cfg.tech_compressor

    # Basic existence checks
    if not hasattr(model, "Balance"):
        raise AttributeError("Model has no Balance constraint. (Expected model.Balance)")
    if not hasattr(model, "dual"):
        raise AttributeError("Model has no dual suffix. (Expected model.dual)")
    if not hasattr(model, "in_frac") or not hasattr(model, "Fe"):
        raise AttributeError("Model missing in_frac/Fe params needed to infer coefficients.")
    if (g, cfg.e_comp) not in model.f_out:
        raise KeyError(f"Compressor tech '{g}' does not export '{cfg.e_comp}' (not in model.f_out).")
    if (g, cfg.e_liq) not in model.f_in:
        raise KeyError(f"Compressor tech '{g}' does not import '{cfg.e_liq}' (not in model.f_in).")
    if (g, cfg.e_el) not in model.f_in:
        # It's possible electricity is not an input in some scenarios
        print(f"⚠️ '{g}' does not import '{cfg.e_el}'. Electricity wedge term will be NaN/0.")
    fe = float(value(model.Fe[g]))
    if fe <= 0:
        raise ValueError(f"Fe[{g}] must be > 0, got {fe}.")

    # Infer inputs per 1 unit of CO2_Comp output:
    # Generation[g,e_comp,t] = out_frac[g,e_comp] * Fuelusetotal[g,t] * Fe[g]
    # For this tech, out_frac typically = 1 for the exported commodity.
    out_frac = float(value(model.out_frac[g, cfg.e_comp])) if hasattr(model, "out_frac") else 1.0
    if out_frac <= 0:
        raise ValueError(f"out_frac[{g},{cfg.e_comp}] must be > 0, got {out_frac}.")

    # 1 unit of CO2_Comp output corresponds to Fuelusetotal = 1/(out_frac*Fe)
    fuelusetotal_per_1_out = 1.0 / (out_frac * fe)

    in_frac_liq = float(value(model.in_frac[g, cfg.e_liq]))
    alpha_liq = in_frac_liq * fuelusetotal_per_1_out  # CO2_Liq input per 1 CO2_Comp output

    if (g, cfg.e_el) in model.f_in:
        in_frac_el = float(value(model.in_frac[g, cfg.e_el]))
        alpha_el = in_frac_el * fuelusetotal_per_1_out  # Electricity input per 1 CO2_Comp output
    else:
        alpha_el = float("nan")

    rows: List[Dict[str, Any]] = []

    for t in model.T:
        comp_out = float(value(model.Generation[g, cfg.e_comp, t]))
        if comp_out <= cfg.run_threshold:
            continue  # only hours where compressor is running

        # Duals -> nodal prices
        d_liq = _dual(model, cfg.area_co2, cfg.e_liq, t)
        d_comp = _dual(model, cfg.area_co2, cfg.e_comp, t)

        p_liq = _price_from_dual(d_liq)
        p_comp = _price_from_dual(d_comp)

        # Electricity nodal price (often DK1)
        d_el = _dual(model, cfg.area_el, cfg.e_el, t)
        p_el = _price_from_dual(d_el)

        # Predicted p_comp from upstream nodes + compressor marginal electricity
        # p_comp_pred ≈ alpha_liq*p_liq + alpha_el*p_el
        # (If alpha_el is nan, prediction will be nan.)
        p_comp_pred = alpha_liq * p_liq + alpha_el * p_el if not pd.isna(alpha_el) else float("nan")

        rows.append({
            "t": str(t),
            "comp_out": comp_out,
            "dual_CO2_Liq": d_liq,
            "dual_CO2_Comp": d_comp,
            "dual_Electricity": d_el,
            "p_CO2_Liq": p_liq,
            "p_CO2_Comp": p_comp,
            "p_Electricity": p_el,
            "alpha_CO2_Liq_per_CO2_Comp": alpha_liq,
            "alpha_Electricity_per_CO2_Comp": alpha_el,
            "p_CO2_Comp_pred": p_comp_pred,
            "pred_error": (p_comp - p_comp_pred) if not pd.isna(p_comp_pred) else float("nan"),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("⚠️ No hours found where compressor is running (comp_out <= threshold for all t).")
        return df

    # Summary stats
    print("\n=== CO2 WEDGE CHECK (compressor-running hours only) ===")
    print(f"Tech: {g}")
    print(f"Area CO2: {cfg.area_co2} | Area Electricity: {cfg.area_el}")
    print(f"Inputs per 1 {cfg.e_comp} out: {cfg.e_liq}={alpha_liq:.6g}, Electricity={alpha_el:.6g}")
    print(f"Rows (hours running): {len(df)}")
    print("\nAverages over running hours:")
    print(df[["p_CO2_Liq", "p_CO2_Comp", "p_Electricity", "pred_error"]].mean(numeric_only=True))

    if cfg.out_csv:
        df.to_csv(cfg.out_csv, index=False)
        print(f"\n→ Wrote wedge diagnostics to: {cfg.out_csv}")

    return df