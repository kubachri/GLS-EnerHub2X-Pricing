import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

# === CONFIGURATION ===
output_path = Path("Scenario_Analysis.xlsx")
scenario_files = {
    "Baseline":       "results/Results_Baseline.xlsx",
    "NoStorageCO2":   "results/Results_NoStorageCO2.xlsx",
    "NoStorageALL":   "results/Results_NoStorageALL.xlsx",
    "H2Sales":        "results/Results_H2Sales.xlsx",
    "CO2Abundance":   "results/Results_CO2Abundance.xlsx"
}
baseline_path = scenario_files["Baseline"]

# === 1. MethanolLDC ===
ldc_data = {}
for name, path in scenario_files.items():
    try:
        df = pd.read_excel(path, sheet_name="ResultT", engine="openpyxl")
        row = df[(df["Result"] == "Operation") &
                 (df["tech"] == "MethanolSynthesis") &
                 (df["energy"] == "Methanol")].drop(columns=["Result", "tech", "energy"])
        if not row.empty:
            ldc_data[name] = row.iloc[0]
    except Exception as e:
        print(f"❌ MethanolLDC: {name} failed - {e}")

df_ldc = pd.DataFrame(ldc_data).T
df_ldc.index.name = "Scenario"

# === 2. CO2Duals ===
co2_data = {}
for name, path in scenario_files.items():
    try:
        df = pd.read_excel(path, sheet_name="Duals", engine="openpyxl")
        df = df[(df["Area"] == "Skive") & (df["Energy"] == "CO2_Liq")]
        if not df.empty:
            df["Hour"] = df["Time"].str.extract(r'T(\d+)').astype(int)
            df.set_index("Hour", inplace=True)
            co2_data[name] = -df["Dual"]
    except Exception as e:
        print(f"❌ CO2Duals: {name} failed - {e}")

df_co2 = pd.DataFrame(co2_data)
df_co2.index.name = "Hour"

# === 3. MethanolMarginal ===
methanol_data = {}
for name, path in scenario_files.items():
    try:
        df = pd.read_excel(path, sheet_name="Duals", engine="openpyxl")
        df = df[df["Area.Fuel"] == "DK1.Methanol"].copy()
        df["Week"] = df["Step"].str.extract(r'Target(\d+)').astype(int)
        df.set_index("Week", inplace=True)
        methanol_data[name] = -df["Dual Value"]
    except Exception as e:
        print(f"❌ MethanolMarginal: {name} failed - {e}")

df_methanol = pd.DataFrame(methanol_data)
df_methanol.index.name = "Week"

# === 4. ObjFunc ===
tracked_elements = [
    "Buy_Electricity", "Buy_NatGas", "Sell_Electricity", "Sell_H2",
    "Variable_OM_WindTurbine", "Variable_OM_ElectricStorage", "Variable_OM_Digester",
    "Variable_OM_BiogasUpgrade", "Variable_OM_CO2Compressor", "Variable_OM_Boiler"
]
obj_data = {}
for name, path in scenario_files.items():
    try:
        df = pd.read_excel(path, sheet_name="ObjDecomp", engine="openpyxl").set_index("Element")
        row = {el: df["Contribution"].get(el, None) for el in tracked_elements}
        row["Total Profit"] = df["Contribution"].get("TotalProfit", None)
        obj_data[name] = row
    except Exception as e:
        print(f"❌ ObjFunc: {name} failed - {e}")

df_obj = pd.DataFrame.from_dict(obj_data, orient="index")
df_obj.index.name = "Scenario"

# === 5. ElectricityPrice (hourly and weekly) ===
try:
    df_elec = pd.read_excel(baseline_path, sheet_name="ResultA", engine="openpyxl")
    df_elec = df_elec[(df_elec["Result"] == "Import_price_EUR") & 
                      (df_elec["energy"] == "Electricity")]
    price_row = df_elec.drop(columns=["Result", "area", "energy"]).iloc[0]
    df_hourly_price = pd.DataFrame({
        "Hour": range(1, len(price_row)+1),
        "ElectricityPrice": price_row.values
    })

    # Compute weekly averages (52 weeks, each 168 hours)
    weekly_avg = df_hourly_price.groupby((df_hourly_price["Hour"] - 1) // 168 + 1).mean()
    weekly_avg.index.name = "Week"
    df_methanol["ElectricityPrice"] = weekly_avg["ElectricityPrice"]
except Exception as e:
    print(f"❌ ElectricityPrice failed - {e}")
    df_hourly_price = pd.DataFrame()

# === WRITE TO EXCEL ===
with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_ldc.to_excel(writer, sheet_name="MethanolLDC")
    df_co2.to_excel(writer, sheet_name="CO2Duals")
    df_methanol.to_excel(writer, sheet_name="MethanolMarginal")
    df_obj.to_excel(writer, sheet_name="ObjFunc")
    if not df_hourly_price.empty:
        df_hourly_price.to_excel(writer, sheet_name="ElectricityPrice", index=False)

print("✅ All sheets updated successfully.")
