import pandas as pd
from pathlib import Path

def compare_results(results_dir: str = "results", base_name: str = "Results.xlsx", test_mode: bool = True):
    """
    Compare centralized vs strategic results.

    Features:
        - Compare Sale and Export_price_EUR from ResultAsum (unchanged)
        - Compare ResultTsum (or ResultT) for all technologies/energies
        - Compute implicit CO2 price (Sale_EUR / Sale)
        - Reconstruct CO2 price using inverse demand
        - Export merged comparison to Excel
    """
    results_dir = Path(results_dir)

    # Determine file names
    if test_mode:
        base_name = "test_" + base_name
    base_file = results_dir / base_name
    strat_file = results_dir / base_file.name.replace(".xlsx", "-strategic.xlsx")

    if not base_file.exists() or not strat_file.exists():
        raise FileNotFoundError(f"Could not find result files in {results_dir}")

    print(f"[INFO] Comparing results:\n  Centralized: {base_file.name}\n  Strategic:   {strat_file.name}")

    # =============================================
    # --- Explore ResultAsum for global results ---
    # =============================================

    # --- Load ResultAsum ---
    df_base_A = pd.read_excel(base_file, sheet_name="ResultAsum")
    df_strat_A = pd.read_excel(strat_file, sheet_name="ResultAsum")

    df_base_A.columns = [c.strip() for c in df_base_A.columns]
    df_strat_A.columns = [c.strip() for c in df_strat_A.columns]

    # --- Detect format ---
    if "energy" in df_base_A.columns:
        base_filt = df_base_A[df_base_A["Result"].isin(["Sale", "Export_price_EUR"])]
        strat_filt = df_strat_A[df_strat_A["Result"].isin(["Sale", "Export_price_EUR"])]
        merge_keys = ["Result", "area", "energy"]
        shared_energies = sorted(set(base_filt["energy"]) & set(strat_filt["energy"]))
    else:
        base_filt = df_base_A[df_base_A["Result"].isin(["Sale", "Export_price_EUR", "Sale_EUR"])].copy()
        strat_filt = df_strat_A[df_strat_A["Result"].isin(["Sale", "Export_price_EUR", "Sale_EUR"])].copy()
        merge_keys = ["Result", "area"]
        shared_energies = sorted(set(base_filt.columns) & set(strat_filt.columns) - set(merge_keys))

    print(f"[INFO] Shared energies found: {shared_energies}")

    # --- Merge and compute numeric deltas Δ ---
    merged_A = pd.merge(
        base_filt, strat_filt,
        on=merge_keys, suffixes=("_central", "_strategic"), how="outer"
    )

    for e in shared_energies:
        if f"{e}_central" in merged_A.columns and f"{e}_strategic" in merged_A.columns:
            merged_A[f"{e}_Δ"] = merged_A[f"{e}_strategic"] - merged_A[f"{e}_central"]

    sale_df = merged_A[merged_A["Result"] == "Sale"]
    price_df = merged_A[merged_A["Result"] == "Export_price_EUR"]

    print("\n[SUMMARY] Sale quantity comparison (Δ = strategic - central):")
    print(sale_df[merge_keys + [c for c in sale_df.columns if c.endswith('Δ')]].to_string(index=False))

    print("\n[SUMMARY] Export price comparison (Δ = strategic - central):")
    print(price_df[merge_keys + [c for c in price_df.columns if c.endswith('Δ')]].to_string(index=False))

    # =============================================
    # --- Explore ResultTsum for global results ---
    # =============================================

    # --- Load ResultTsum ---
    try:
        df_base_Tsum = pd.read_excel(base_file, sheet_name="ResultTsum")
        df_strat_Tsum = pd.read_excel(strat_file, sheet_name="ResultTsum")
        df_base_Tsum.columns = [c.strip() for c in df_base_Tsum.columns]
        df_strat_Tsum.columns = [c.strip() for c in df_strat_Tsum.columns]
    except Exception as e:
        print(f"[ERROR] Could not read ResultTsum: {e}")
        df_base_Tsum = df_strat_Tsum = pd.DataFrame()

    # --- Merge and compute numeric deltas Δ ---
    try:
        merge_keys_T = ["Result", "tech"]
        merged_T = pd.merge(
            df_base_Tsum, df_strat_Tsum,
            on=merge_keys_T,
            suffixes=("_central", "_strategic"),
            how="outer"
        )

        # Automatically detect all columns to compare
        all_cols = df_base_Tsum.columns.tolist()
        compare_cols = [c for c in all_cols if c not in merge_keys_T]

        print("\n[INFO] ResultTsum comparison:")
        for col in compare_cols:
            col_central = f"{col}_central"
            col_strategic = f"{col}_strategic"
            if col_central in merged_T.columns and col_strategic in merged_T.columns:
                merged_T[f"{col}_Δ"] = merged_T[col_strategic] - merged_T[col_central]

                # Print summary statistics
                delta = merged_T[f"{col}_Δ"]
                print(f"Column: {col}")
                print(f"  Max Δ: {delta.max():.3f}, Min Δ: {delta.min():.3f}, Mean Δ: {delta.mean():.3f}\n")

    except Exception as e:
        print(f"[WARN] Could not compare ResultTsum: {e}")
        merged_T = pd.DataFrame()

    # --- Compute implicit and reconstructed CO2 price -----
    # TODO: Implement if needed

    # ======================================================
    # --- Export merged comparisons to Excel --------------
    # ======================================================
    # if not merged_T.empty:
    #     print("[INFO] ResultTsum comparison finished.")
    #     total_changes = merged_T[[c for c in merged_T.columns if "_Δ" in c]].abs().sum().sum()
    #     if total_changes == 0:
    #         print("No differences detected between central and strategic runs.")
    #     else:
    #         print(f"Total absolute difference across all columns: {total_changes:.3f}")
    # else:
    #     print("[WARN] ResultTsum comparison could not be performed.")

    # # --- Export ---
    # try:
    #     merged_T.to_excel("ResultTsum_comparison.xlsx", index=False)
    #     print("[INFO] Comparison exported to 'ResultTsum_comparison.xlsx'")
    # except Exception as e:
    #     print(f"[WARN] Could not export ResultTsum comparison: {e}")
    

    # output_path = results_dir / "Results_comparison.xlsx"
    # with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    #     merged.to_excel(writer, sheet_name="KeyComparisons", index=False)
    #     if implicit_price is not None or reconstructed_price is not None:
    #         summary = pd.DataFrame(
    #             [{
    #                 "Implicit_CO2_price": implicit_price,
    #                 "Reconstructed_CO2_price": reconstructed_price,
    #                 "a_CO2": a_co2,
    #                 "b_CO2": b_co2
    #             }]
    #         )
    #         summary.to_excel(writer, sheet_name="CO2_Price_Summary", index=False)
    # print(f"\n[INFO] Comparison exported: {output_path.resolve()}")


compare_results()