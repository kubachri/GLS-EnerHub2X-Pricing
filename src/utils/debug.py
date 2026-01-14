import pandas as pd
def debug_carriermix(in_frac, out_frac):
    # 1) Gather only the entries you know exist
    records = []
    for (g,f), v in out_frac.items():
        records.append({
            'tech':      g,
            'direction': 'Export',
            'carrier':   f,
            'value':     round(v,4)
            })
    for (g,f), v in in_frac.items():
        # in_frac was your dict[(g,f)] = v (only positives)
        records.append({
            'tech':      g,
            'direction': 'Import',
            'carrier':   f,
            'value':     round(v,4)
        })

    # 2) Build a flat DataFrame
    df = pd.DataFrame.from_records(records)

    # 3) Pivot so carriers become columns, blanking missing cells
    df_pivot = (
        df
        .pivot(index=['tech','direction'], columns='carrier', values='value')
        .fillna('')
    )

    # 4) Write to Excel
    df_pivot.to_excel('in_out_frac.xlsx', sheet_name='σ-fractions')
    print("► wrote in_out_frac.xlsx")

def debug_fuels(model):
    # Collect debug info -- NEED UPDATE NEVER GETS CALLED
    rows = []
    for g in model.G:
        name = g.lower()
        if 'wind' in name or 'solar' in name:
            # find the electricity export (g,'Electricity') parameters
            # you could generalize if your out_frac key is different
            if ('Electricity' not in [e for (gg, e) in model.f_out if gg == g]):
                continue
            out_frac = model.out_frac[g, 'Electricity']
            print('out frac', out_frac)
            eff = model.Fe[g]

            for idx, t in enumerate(model.T):
                if idx >= 25:  # only hours 1–100
                    break
                cap = model.capacity[g]
                prof = model.Profile[g, t]
                max_fuel = cap * prof  # max pre‐efficiency fuel
                max_el = max_fuel * eff * out_frac  # what that fuel could produce
                fuel_act = model.Fuelusetotal[g, t].value  # actual fuel used
                gen_act = model.Generation[g, 'Electricity', t].value
                efficiency = model.Fe[g
                ]
                rows.append({
                    'tech': g,
                    'time': t,
                    'cap*profile': max_fuel,
                    'fuel_used': fuel_act,
                    'max_electricity': max_el,
                    'actual_electricity': gen_act,
                    'efficiency': efficiency
                })

    # Build DataFrame and display or save
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # Or: df.to_csv('wind_solar_debug.csv', index=False)
    print()