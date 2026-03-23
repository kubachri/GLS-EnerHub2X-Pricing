import pandas as pd
from pyomo.core.base.param import Param
from pyomo.core.base.set import Set
from pyomo.environ import value
from pathlib import Path
import numpy as np


def export_inputs(model, cfg, path: str = None):
    """
    Export all Sets and Params of `model` into an Excel workbook.
    """
    if path is None:
        project_root = Path(__file__).parents[2]
        filename = "Inputs.xlsx"
        if cfg.test_mode:
            filename = "test_" + filename
        if cfg.strategic:
            filename = filename.replace(".xlsx", "-strategic.xlsx")
        out = project_root / "results" / filename
    else:
        out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer = pd.ExcelWriter(out, engine='xlsxwriter')
    except PermissionError:
        for i in range(1, 100):
            trial = out.with_name(f"{out.stem}({i}){out.suffix}")
            try:
                writer = pd.ExcelWriter(trial, engine='xlsxwriter')
                print(f"⚠️  {out.name} is in use—trying {trial.name}")
                out = trial  # Update the output path
                break
            except PermissionError:
                continue
        else:
            raise PermissionError(f"Could not write to {out} or any fallback after 100 attempts.")


    sheet_data = {}

    # Export all Sets to a single sheet, side-by-side
    set_dfs = []
    set_names = []

    for s in model.component_objects(Set, descend_into=True):
        rows = [(member if isinstance(member, tuple) else (member,)) for member in s]
        if not rows:
            continue
        dim = len(rows[0])
        col_names = [f"{s.name}_idx{i+1}" for i in range(dim)]
        df = pd.DataFrame(rows, columns=col_names)
        set_dfs.append(df)
        set_names.append(s.name)

    # Concatenate all set DataFrames with spacing columns in between
    from itertools import zip_longest

    max_rows = max(df.shape[0] for df in set_dfs)
    padded_dfs = []

    for df in set_dfs:
        padded = df.reindex(range(max_rows))  # pad shorter DataFrames
        padded_dfs.append(padded)

    # Add 1 blank column between each
    combined = padded_dfs[0]
    for df in padded_dfs[1:]:
        spacer = pd.DataFrame([""] * max_rows, columns=[""])
        combined = pd.concat([combined, spacer, df], axis=1)

    # Write to single sheet "Sets"
    # combined.to_excel(writer, sheet_name="Sets", index=False)
    sheet_data["Sets"] = combined

    tech_params = {
    "capacity", "original_capacity", "Fe", "soc_init",
    "soc_max", "cstart", "cvar", "RampRate", "Minimum"
    }
    tech_df = {}
    # Export Params
    for p in model.component_objects(Param, descend_into=True):
        pname = p.name
        rows = []

        # ✅ Only iterate over defined keys
        for idx in p:
            idx_tup = idx if isinstance(idx, tuple) else (idx,)
            try:
                v = value(p[idx_tup])
            except:
                v = np.nan
            rows.append(idx_tup + (v,))

        if not rows:
            continue

        if pname in tech_params:
            df = pd.DataFrame(rows, columns=["Tech", pname])
            if "Tech" in tech_df:
                tech_df["Tech"] = pd.merge(tech_df["Tech"], df, on="Tech", how="outer")
            else:
                tech_df["Tech"] = df
            continue

        # Special reshape for price_buy / price_sale
        if pname in {"price_buy", "price_sale", "InterconnectorCapacity", "demand"}:
            df = pd.DataFrame(rows, columns=["Area", "Fuel", "Time", pname])
            df = df[df[pname].notna()]  # ✅ filter only defined nonzero
            df["Area.Fuel"] = df["Area"] + "." + df["Fuel"]
            df_pivot = df.pivot(index="Time", columns="Area.Fuel", values=pname)
            df_pivot.reset_index(inplace=True)
            df_pivot.rename(columns={"Time": "Hour"}, inplace=True)
            short_name = pname if len(pname) <= 25 else pname[:25]
            sheet_data[f"{short_name}"] = df_pivot

            continue

        if pname == "Profile":
            df = pd.DataFrame(rows, columns=["Tech", "Time", pname])
            df = df[df[pname].notna()]  # ✅ include zero values, exclude only NaN
            df_pivot = df.pivot(index="Time", columns="Tech", values=pname)
            df_pivot.reset_index(inplace=True)
            df_pivot.rename(columns={"Time": "Hour"}, inplace=True)
            short_name = pname if len(pname) <= 25 else pname[:25]
            sheet_data[f"{short_name}"] = df_pivot
            continue

        if pname in {"in_frac", "out_frac"}:
            df = pd.DataFrame(rows, columns=["Tech", "Fuel", pname])
            df = df[df[pname].notna()]
            df_pivot = df.pivot(index="Tech", columns="Fuel", values=pname)
            df_pivot.reset_index(inplace=True)
            short_name = pname if len(pname) <= 25 else pname[:25]
            sheet_data[f"{short_name}"] = df_pivot
            continue

        # Generic param export
        num_idx = len(rows[0]) - 1
        col_names = [f"{pname}_idx{i+1}" for i in range(num_idx)] + [pname]
        df = pd.DataFrame(rows, columns=col_names)
        sheet_data[f"Param__{pname}"] = df

    if "Tech" in tech_df:
        # tech_df["Tech"].to_excel(writer, sheet_name="tech_df", index=False)
        sheet_data["tech_df"] = tech_df["Tech"]

    # Write in the specified order
    sheet_order = [
        "Sets", "tech_df", "in_frac", "out_frac", "Profile",
        "demand", "price_buy", "price_sale", "InterconnectorCapacity"
    ]
    for sheet in sheet_order:
        if sheet in sheet_data:
            sheet_data[sheet].to_excel(writer, sheet_name=sheet, index=False)

    # Write all remaining sheets not in the priority list
    for sheet, df in sheet_data.items():
        if sheet not in sheet_order:
            df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    print("Input data exported successfully.")
    print(f"File: {out}")