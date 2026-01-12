# src/strategic/submodel_methanol.py

import pyomo.environ as pyo
from copy import deepcopy

from src.data.loader import load_data
from src.data.preprocess import scale_tech_parameters, slice_time_series
from src.model.sets import define_sets
from src.model.parameters import define_params
from src.model.variables import define_variables
from src.model.constraints import add_constraints
from src.model.objective import define_objective


def build_methanol_model(cfg, supply, price_co2=None, techs=["CO2Compressor", "CO2Storage", "MethanolSynthesis"], co2_label='CO2'):
    """
    Build a restricted Pyomo model for the methanol actor.

    Parameters
    ----------
    cfg : ModelConfig
        Full system configuration, but will be partially overridden.
    supply : dict
        CO2 supply at each time step.
        Format: {t: quantity, ...}
    price_co2 : dict or None
        Optional time-dependent CO2 price provided by the biogas submodel.
        Format: {t: price_t, ...}. If None, CO2 price is set to zero.
    techs : list of str
        Names of technologies to include (default: ["MethanolSynthesis"]).
    co2_label : str
        Fuel name used in the model (default: 'CO2').

    Returns
    -------
    model : ConcreteModel
    """

    # ------------------------------------------------------------------
    # 1. Load full data and restrict to methanol technologies only
    # ------------------------------------------------------------------
    data, tech_df = load_data(cfg)

    data = deepcopy(data)  # do not mutate original session memory
    tech_df = deepcopy(tech_df)

    data, tech_df = scale_tech_parameters(data, tech_df)

    if cfg.test_mode:
        data = slice_time_series(data, cfg.n_test)

    print("\nAll data loaded for methanol submodel.\n")
    
    # Restrict technologies G
    data['G'] = [g for g in data['G'] if g in techs]

    # Ensure CO2 is in the fuels set
    if co2_label not in data['F']:
        data['F'].append(co2_label)

    # Ensure price_buy has entries for CO2
    area_import = "DK1"  # Assuming import area is DK1; adjust as needed

    if 'price_buy' not in data:
        data['price_buy'] = {}
    for t in data['T']:
        key = (area_import, co2_label, t)
        if price_co2 is not None:
            data['price_buy'][key] = price_co2.get(t, 0.0)
        else:
            data['price_buy'][key] = 0.0

    # ------------------------------------------------------------------
    # 2. Assemble Pyomo model
    # ------------------------------------------------------------------
    m = pyo.ConcreteModel()

    #DemandTarget
    m.Demand_Target = cfg.demand_target
    if m.Demand_Target:
        print("Running with a demand target ...\n")

    #Green fuels
    m.GreenElectricity = cfg.green_electricity
    if m.GreenElectricity:
        print("Running with a green electrity from the grid (<20 EUR/MWh) ...\n")

    #Electricity mandate
    m.ElectricityMandate = cfg.electricity_mandate
    if m.ElectricityMandate:
        print(f"Running with an electricity mandate of {m.ElectricityMandate*100}% ...\n")

    #Electricity export limit
    m.ElProdToGrid = cfg.el_prod_to_grid
    if m.ElProdToGrid:
        print(f"Running with grid export to production ratio of {m.ElProdToGrid*100}% ...\n")

    define_sets(m, data)
    define_params(m, data, tech_df)
    define_variables(m)
    add_constraints(m)
    define_objective(m, cfg=cfg)

    # ------------------------------------------------------------------
    # 3. Limit CO2 supply (availability constraint)
    # ------------------------------------------------------------------
    # CO2 is available from biogas
    def co2_supply_limit_rule(m, t):
        return m.Buy[area_import, co2_label, t] <= supply.get(t, 0.0)
    m.CO2_SupplyLimit = pyo.Constraint(m.T, rule=co2_supply_limit_rule)

    # # Generation of CO2 is limited by the supplied quantity
    # def co2_generation_limit_rule(m, t):
    #     return sum(m.Generation[g, co2_label, t] for g in m.G if (g, co2_label) in m.TechToEnergy) <= supply.get(t, 0.0)
    # m.CO2_GenerationLimit = pyo.Constraint(m.T, rule=co2_generation_limit_rule)

    return m
