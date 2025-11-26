# src/data/sensitivity.py

def apply_sensitivity_overrides(tech_df, data):
    """
    Modify tech_df and/or data dictionaries in-place based on desired sensitivity case.
    """
    # Example override
    # if 'CO2Storage' in tech_df.index:
    #     tech_df.at['CO2Storage', 'VariableOmcost'] = 100

    # You can add more changes:
    # tech_df.at['Electrolyzer', 'StartupCost'] = 50000
    # data['price_buy'][('DK1', 'Electricity', 'T001')] = 200
    # tech_df.at['ElectricStorage', 'InitialVolume'] = 1
    # tech_df.at['CO2Storage', 'InitialVolume'] = 1
    # tech_df.at['H2Storage', 'InitialVolume'] = 1

    # tech_df.at['CO2Liquefaction', 'Capacity'] = 3.188356164
    # tech_df.at['CO2Liquefaction', 'Capacity'] = 3.188356164

    tech_df.at['CO2Compressor', 'Capacity'] = 3.188356164
    tech_df.at['CO2Compressor', 'Capacity'] = 3.188356164

    # tech_df.at['H2Storage', 'Capacity'] = 5
    # tech_df.at['H2Storage', 'StorageCap'] = 5
    




    
    # tech_df.at['ElectricStorage', 'StorageCap'] = 10
    
    return tech_df, data