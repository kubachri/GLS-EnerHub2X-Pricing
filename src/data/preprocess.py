# src/data/preprocess.py

def scale_tech_parameters(data, tech_df):
    """
    1) Scale the raw capacities, minima, and ramp‐rates
    2) Return UC and RR sets (lists of technologies)
    3) Return the final capacity dict
    """
    G_s       = data['G_s']
    sigma_in  = data['sigma_in']
    sigma_out = data['sigma_out']

    # Assign storage capacity directly from StorageCap
    tech_df.loc[G_s, 'Capacity'] = tech_df.loc[G_s, 'StorageCap']

    # Keep copies of your originals
    orig_cap  = tech_df['Capacity'].copy()
    orig_min  = tech_df['Minimum'].copy()
    orig_ramp = tech_df['RampRate'].copy()

    # Sum of all inputs per technology
    sum_in_raw = {
        g: sum(v for (gg, e), v in sigma_in.items() if gg == g)
        for g in tech_df.index
    }

    # Which technologies have nonzero minimum or ramp?
    UC = [g for g in tech_df.index if orig_min[g]  > 0 or orig_ramp[g] > 0]

    # Now scale your DataFrame in place…
    for g in UC:
        tech_df.at[g, 'Minimum'] = sum_in_raw[g] * orig_cap[g] * orig_min[g]
        tech_df.at[g, 'RampRate'] = sum_in_raw[g] * orig_cap[g] * orig_ramp[g]
    for g in tech_df.index:
        newc = sum_in_raw[g] * orig_cap[g]
        tech_df.at[g, 'Capacity'] = newc

    # And emit the Python dict your Param() will consume
    capacity = {g: tech_df.at[g, 'Capacity'] for g in tech_df.index}

    # Stick UC, RR and capacity back into your data dict for easy access
    data['UC']       = UC
    data['capacity'] = capacity
    data['original_cap'] = orig_cap

    return data, tech_df

def slice_time_series(data, n_hours):
    """
    Truncate all time‐series in `data` to the first n_hours periods.
    Also replaces data['T'] with the shortened time‐set.
    """
    # 1) Get the full ordered list of time points
    T_all = list(data['T'])
    # 2) Slice it
    T_short = T_all[:n_hours]

    # 3) Helper to keep only entries whose last key component is in T_short
    def _keep(d):
        return {k: v for k, v in d.items() if k[-1] in T_short}

    # 4) Apply to each time‐series dict
    data['Profile']    = _keep(data['Profile'])
    data['Demand']     = _keep(data['Demand'])
    data['price_buy']  = _keep(data['price_buy'])
    data['price_sell'] = _keep(data['price_sell'])
    data['Xcap']       = _keep(data['Xcap'])
    

    # 5) Override the time‐set itself
    data['T'] = T_short

    # 6) Trim DemandTarget to only include active steps (Target1, Target2, ..., TargetN)
    num_weeks = (len(T_short) + 167) // 168  # ceiling division to cover partial weeks
    active_steps = {f"Target{i+1}" for i in range(num_weeks)}

    data['DemandTarget'] = {
                (step, af): val
                for (step, af), val in data['DemandTarget'].items()
                if step in active_steps
            }
    
    #Only keep the relevant weeks
    data['WeekOfT'] = {t: data['WeekMap'][t] for t in data['T']}
    return data