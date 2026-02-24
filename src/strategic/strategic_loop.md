**FUNCTION**        
***run_cournot***(cfg, tol, max_iter, damping):

    # --- Initialization phase ---
    solve(base_model)    # centralized configuration
    strategic_suppliers, strategic_demanders ← extract_strategic_actors(cfg)
    strategies ← extract_current_state(base_model, strategic_suppliers, strategic_demanders) # CO2 supply, demand, and dual prices

    # --- Iteration phase (best-response loop) ---
    FOR iteration in 1..max_iter:
        max_change ← 0

        FOR each strategic_supplier i:
            fix_competitors(supply_model, strategies, excluding=i)
            solve(supply_model, strategies[strategic_demanders])  # take as input the demanders strategies: 

            # Update supplier i sales (to fix for other suppliers profit maximization)
            q_i_new ← extract_sales_for_supplier(supply_model, i)
            q_i_updated ← damping * q_i_new + (1 - damping) * q_i_old
            strategies[i] ← q_i_updated   # CO2 supply

            # Compute maximal change in results to assess convergence
            max_change ← max(max_change, |q_i_updated - q_i_old|)

        FOR each strategic_demander j:
            fix_competitors(demand_model, strategies, excluding=j)
            solve(demand_model, strategies[strategic_suppliers])  # take as input the suppliers strategies: CO2 supply
            strategies[j] ← demand_model  # CO2 demand and dual prices

        PRINT iteration summary (max_change)
        IF max_change < tol: break (converged)

    # --- Finalization phase ---
    final_model ← build_model(cfg)
    fix_all_strategic_sales(final_model, strategies)
    solve(final_model)
    export_results()
    
    return final_model, strategies


### Cournot Best-Response Iteration Algorithm (CO2 Market)

**Input:** Model configuration `cfg`, tolerance `tol`, maximum iterations `max_iter`, damping factor `damping`, strategic commodity `co2_label`  
**Output:** Converged strategies for CO2 suppliers and demanders, iteration log

---

1. **Initialization**
   1. Solve the centralized (full) model to obtain CO2 supply, demand, and dual prices.
   2. Identify strategic actors:
      - Suppliers (e.g., BiogasUpgrade)
      - Demanders (e.g., MethanolSynthesis, CO2Storage)
   3. Initialize supply strategies using centralized CO2 generation.
   4. Initialize demand strategies by constructing initial demand-price blocks for each time step:
      - Include external market fallback block (low price - large quantity).
      - Include internal demand derived from centralized CO2 fuel use and dual prices (for willingness to pay).

2. **Best-Response Iteration**
   Repeat until convergence or `max_iter` reached:
   
   1. **Supply-Side Optimization**
      - Solve each strategic supplier submodel (e.g., biogas) given current *observed* demand-price blocks.
      - Extract optimal CO2 sales (`co2_sell`).
      - Compute available supply to downstream actors (e.g., methanol plant).
      - Update supplier strategies using damping:
        ```
        curr[t] = damping * old_value + (1 - damping) * new_value
        ```

   2. **Demand-Side Optimization**
      - Solve each strategic demander submodel (e.g., methanol) given updated supplier strategies.
      - Extract internal CO2 use and dual prices.
      - Update demand-price blocks for next iteration using new internal demand and dual prices.
   
   4. **Check Convergence**
      - Track maximum strategy change across time steps (`max_change`).
      - If `max_change < tol`, declare convergence and exit loop.

3. **Finalization**
    Log iteration results (strategies, prices, objectives).

