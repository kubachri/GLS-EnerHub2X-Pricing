# src/data/loader.py

import os
import pandas as pd
import numpy as np
from src.model.sensitivities import apply_sensitivity_overrides
from src.utils.assign_hours_to_weeks import build_full_year_week_map

def load_data(cfg):
    """
    Load all data for the Pyomo model from a single Excel workbook
    NewDataFormat.xlsx located at the project root.
    Returns the same `data` dict shape as before.
    """
    # Path to the Excel file (assumed in project root)
    # Choose whether to run a single scenario or multiple scenarios one after the other
    excel_path = cfg.data_file

    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"Could not find Excel data file: {excel_path}")

    # Load all sheets
    xls = pd.ExcelFile(excel_path)
    sheets = {name: xls.parse(name) for name in xls.sheet_names}

    # -----------------------
    # 1) TECHNOLOGIES (G)
    # -----------------------
    # TechsIncluded: no header, names in col A
    # Read the TechsIncluded sheet without treating any row as header
    techs_included = xls.parse('TechsIncluded', header=None)
    # Now column 0 is truly all your tech names, starting with the very first row
    techs = techs_included[0].dropna().astype(str).tolist()

    # -----------------------
    # 2) LOCATION (area, tech)
    # -----------------------
    loc_df = sheets['Location'].astype(str).dropna(how='all')
    location = list(loc_df.itertuples(index=False, name=None))

    # -----------------------
    # 3) FLOWSET (area_from, area_to, fuel)
    # -----------------------
    flow_df = sheets['Flowset'].astype(str).dropna(how='all')
    flowset = list(flow_df.itertuples(index=False, name=None))

    # Derive A = all unique areas seen in Flowset
    areas = sorted(
        set(flow_df['AreaFrom']).union(flow_df['AreaTo'])
    )


    # 4) CARRIER MIX (sigma_in, sigma_out)
    # ------------------------------------
    # 1) Load raw sheet with no header inference
    cm_raw = xls.parse('Carriermix', header=None)

    # 2) Row index 3 (4th Excel row) has the true column labels
    raw_header = cm_raw.iloc[3].fillna('').astype(str).tolist()
    raw_header[0] = 'tech'    # rename the blank first column

    # 3) Data starts at row index 4
    cm_data = cm_raw.iloc[4:].copy()
    cm_data.columns = raw_header

    # 4) Keep only the rows for your selected techs
    cm_data = cm_data[cm_data['tech'].isin(techs)]

    # 5) Identify which Import/Export cols actually have data
    import_cols = [
        col for col in cm_data.columns
        if col.startswith('Import.') 
        and cm_data[col].notna().any() 
        and (cm_data[col] != 0).any()
    ]
    export_cols = [
        col for col in cm_data.columns
        if col.startswith('Export.') 
        and cm_data[col].notna().any() 
        and (cm_data[col] != 0).any()
    ]

    # 6) Build dictionaries, filtering out any NaN or zero on the fly
    sigma_in = {
        (row.tech, col.split('.',1)[1]): float(v)
        for _, row in cm_data.iterrows()
        for col in import_cols
        for v in [row[col]]
        if pd.notna(v) and v != 0
    }

    sigma_out = {
        (row.tech, col.split('.',1)[1]): float(v)
        for _, row in cm_data.iterrows()
        for col in export_cols
        for v in [row[col]]
        if pd.notna(v) and v != 0
    }

    # Derive F = fuels that have any non-zero import/export for included techs
    fuels = sorted({
        fuel
        for (t,fuel), v in {**sigma_in, **sigma_out}.items()
        if v != 0
    })



    # -----------------------
    # 5) TECHDATA (tech parameters)
    # -----------------------
    
    # 1) Read the sheet with no header inference
    td_raw = xls.parse('Techdata', header=None)

    # 2) Drop any fully blank rows and reset the index
    td_raw = td_raw.dropna(how='all').reset_index(drop=True)

    # 3) Auto-detect which row is the real header:
    #    Look for the first row that contains the string "Capacity"
    header_idx = next(
        i for i, row in td_raw.iterrows()
        if row.astype(str).str.contains('Capacity', case=False).any()
    )

    # 4) Extract and clean the header names
    headers = td_raw.iloc[header_idx].fillna('').astype(str).tolist()
    # The first cell in that row is blank—rename it "tech"
    headers[0] = 'tech'

    # 5) Slice off the header+meta rows, assign our cleaned headers
    tech_df = td_raw.iloc[header_idx+1 :].copy()
    tech_df.columns = headers

    # 6) Keep only the rows whose "tech" is in your TechsIncluded list
    tech_df = tech_df[tech_df['tech'].isin(techs)].set_index('tech')
    tech_df = tech_df.infer_objects(copy=False).fillna(0)
    tech_df = tech_df.infer_objects(copy=False).fillna(0)

    # 8) Identify your storage technologies G_s
    G_s = tech_df.index[tech_df['StorageCap'] > 0].tolist()

    # -----------------------
    # 6) PROFILE & time-index T
    # -----------------------

    # 1) Read the Profile sheet, using the first row as header
    prof_df = xls.parse('Profile', header=0)

    # 2) Make sure the first column is named “Hour”
    prof_df = prof_df.rename(columns={prof_df.columns[0]: 'Hour'})

    # 3) Extract your time‐index in sheet order
    T = prof_df['Hour'].astype(str).tolist()

    # pick only the columns for the included technologies
    tech_cols = [c for c in prof_df.columns[1:] if c in techs]
    prof_df = prof_df[['Hour'] + tech_cols]

    # build Profile only over those
    Profile = {
        (tech, hr): float(row[tech])
        for _, row in prof_df.iterrows()
        for tech in tech_cols
        for hr in [row['Hour']]
    }

    # -----------------------
    # 7) DEMAND (area.energy × time)
    # -----------------------

    # Parse with header=3 to use the 4th row as column names and drop the top 3 metadata rows
    dem_df = xls.parse('DemandHourly', header=3).dropna(how='all')

    # Rename first column to "Hour"
    dem_df.rename(columns={dem_df.columns[0]:'Hour'}, inplace=True)

    # 2) Pick only the columns whose fuel is in your fuels list
    demand_cols = [col for col in dem_df.columns[1:]
               if col.split('.',1)[1] in fuels]

    # 3) Slice to smaller DF
    dem_df = dem_df[['Hour'] + demand_cols]

    # Now exactly like Profile: build (area, energy, hr) → value
    Demand = {
        (area, energy, hr): float(row[col])
        for _, row in dem_df.iterrows()
        for col in demand_cols
        for area, energy in [col.split('.',1)]
        if pd.notna(row[col])
        for hr in [row['Hour']]
    }


    # -----------------------
    # 8) PRICE (import/export)
    # -----------------------

    # 1) Read raw Price sheet without headers
    pr_raw = xls.parse('Price', header=None)

    # 2) Drop fully empty rows
    pr_raw = pr_raw.dropna(how='all').reset_index(drop=True)

    # 3) Auto-detect header row: first row containing "Hour"
    header_idx = next(
        i for i, row in pr_raw.iterrows()
        if row.astype(str).str.strip().eq('Hour').any()
    )

    # 4) Build header
    headers = pr_raw.iloc[header_idx].astype(str).tolist()

    # 5) Slice data below header
    pr_df = pr_raw.iloc[header_idx + 1:].copy()
    pr_df.columns = headers

    # 6) Keep only real time rows (Hour starts with "T")
    pr_df = pr_df[
        pr_df['Hour'].astype(str).str.startswith('T')
    ].reset_index(drop=True)

    # 7) Keep only price columns for fuels in the model
    price_cols = [
        col for col in pr_df.columns[1:]
        if isinstance(col, str)
        and '.' in col
        and col.split('.', 2)[1] in fuels
    ]

    pr_df = pr_df[['Hour'] + price_cols]

    # 8) Build price dictionaries
    price_buy = {}
    price_sell = {}

    for _, row in pr_df.iterrows():
        hr = row['Hour']
        for col in price_cols:
            area, energy, direction = col.split('.', 2)
            val = row[col]
            if pd.isna(val):
                continue
            if direction == 'Import':
                price_buy[(area, energy, hr)] = float(val)
            elif direction == 'Export':
                price_sell[(area, energy, hr)] = float(val)
    # # Apply carbon tax to electricity imports (120 gCO2eq/kWh in 2024)
    # price_buy = {
    #     (area, energy, time): (price + 0.12*cfg.carbon_tax if energy == "Electricity" else price)
    #     for (area, energy, time), price in price_buy.items()
    # }

    # Apply carbon tax to NG usage (198 kgCO2eq/MWh and 50 EUR/tCO2 - 2030 Denmark)
    price_buy = {
        (area, energy, time): (price + 0.198*50 if energy == "NatGas" else price)
        for (area, energy, time), price in price_buy.items()
    }

    # price_sell = {
    #     (area, energy, time): (cfg.carbon_tax if energy == "CO2Comp" else price)
    #     for (area, energy, time), price in price_sell.items()
    # }

    # -----------------------
    # 9) INTERCONNECTOR CAPACITY
    # -----------------------
    # 1) Read so that the 4th Excel row (index 3) is your header
    ic_df = xls.parse('InterconnectorCapacity', header=3) \
            .dropna(how='all') \
            .reset_index(drop=True)

    # 2) Make sure the first column is named “Hour”
    ic_df.rename(columns={ic_df.columns[0]: 'Hour'}, inplace=True)

    # 3) Pick only the columns whose energy is in our fuels list
    ic_cols = [
        col for col in ic_df.columns[1:]
        if col.split('.',1)[1] in fuels
    ]

    # 4) Slice to a smaller DataFrame for inspection
    ic_df = ic_df[['Hour'] + ic_cols]

    # 5) Build Xcap dict from that filtered frame
    Xcap = {
        (area, energy, hr): float(row[col])
        for _, row in ic_df.iterrows()
        for col in ic_cols
        for area, energy in [col.split('.',1)]
        if pd.notna(row[col])
        for hr in [row['Hour']]
    }
    
    location = [(a,t) for (a,t) in location if t in techs]

    flowset = [(a1,a2,f) for (a1,a2,f) in flowset if f in fuels]

    # --- DEMAND TARGET sheet ---
    # --- Load DemandTarget sheet ---
    df = sheets['DemandTarget'].dropna(how='all')

    # Row 3 (index 2) is the actual header row: ["Steps", "DK1.Methanol", ...]
    df.columns = df.iloc[2]
    df = df.drop(index=range(3))  # Drop rows 0–2

    # Rename first column to "Step"
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Step"})

    # Ensure "Step" column is string
    df["Step"] = df["Step"].astype(str).str.strip()

    # Build dictionary for all (step, area.fuel) pairs
    demand_target = {}
    for _, row in df.iterrows():
        step = row["Step"]
        for col in df.columns:
            if col == "Step":
                continue
            val = row[col]
            if pd.notna(val):
                area_fuel = str(col).strip()  # e.g., "DK1.Methanol"
                try:
                    demand_target[(step, area_fuel)] = float(val)
                except Exception as e:
                    print(f"⚠ Skipping ({step}, {area_fuel}) → {val}: {e}")
                    
    # -----------------------
    # Assemble and return
    # -----------------------
    data = {
        'G':            techs,
        'A':            areas,
        'F':            fuels,
        'G_s':          G_s,
        'T':            T,
        'sigma_in':     sigma_in,
        'sigma_out':    sigma_out,
        'Profile':      Profile,
        'Demand':       Demand,
        'price_buy':    price_buy,
        'price_sell':   price_sell,
        'Xcap':         Xcap,
        'FlowSet':      flowset,
        'location':     location,
        'DemandTarget': demand_target
    }

    #Assign all hours to weeks
    data['WeekMap'] = build_full_year_week_map(T)
    data['WeekOfT'] = {t: data['WeekMap'][t] for t in data['T']}

    if cfg.sensitivity:
        tech_df, data = apply_sensitivity_overrides(tech_df, data)

    return data, tech_df