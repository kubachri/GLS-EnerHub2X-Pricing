# src/strategic/submodel_methanol.py
'''
Docstring for EnerHub2X.src.strategic.submodel_methanol

PSEUDO CODE - build_methanol_model() Function Overview
=====================================================

PURPOSE:
  Build a restricted Pyomo optimization model for the methanol actor.
  Extends the centralized base model with limited CO2 exogenous internal supply and
  unlimited external market availability. Limits cost minimization to methanol-specific
  technologies. 

ALGORITHM:

1. MODEL BUILDING
    - Load full data and preprocess
    - Create model and define sets, parameters, variables, constraints
   
2. CO2 SUPPLY DEFINITION [CUSTOM - METHANOL SPECIFIC]
    - Define a CO2 price to enable imports from external market
    - Exogenous CO2 supply input extracted from biogas submodel
    - CO2 internal generation is limited to the supplied amount (announced from biogas submodel)
      >> internal generation is usually converging towards the supplied amount (for optimality)
    
3. CO2 BALANCE CONSTRAINT [CUSTOM - METHANOL SPECIFIC]
    - Override generic CO2 balance constraint
    - Specifics for methanol submodel: internal generation + external buy = co2 demand
    - Dual CO2 price value is extracted from this constraint in post-processing

4. CUSTOM METHANOL PROFIT OBJECTIVE [CUSTOM - METHANOL SPECIFIC]
    - Cost minimization (because methanol does not generate any revenue)
    - Specifically import costs are taking into account CO2 imports

   MINIMIZE: imp_cost + var_om + startup + cfg.penalty * slack_sum

   Scope: restricted to methanol plant technologies only (CO2Compressor, CO2Storage, MethanolSynthesis)

DEVIATIONS FROM CENTRALIZED BASE MODEL:
----------------------------------------
✓ CO2 supply (internal generation) is limited exogenously (vs. internal free flows)
✓ CO2 can be imported from external market at a fixed price (vs. no imports available)
✓ Dual price is dependent of a different balance constraint
✓ Objective function is limited to methanol actors
'''

import pyomo.environ as pyo
from copy import deepcopy

from src.data.loader import load_data
from src.data.preprocess import scale_tech_parameters, slice_time_series
from src.model.sets import define_sets
from src.model.parameters import define_params
from src.model.variables import define_variables
from src.model.constraints import add_constraints
from src.model.objective import define_objective


def build_methanol_model(cfg, co2_supply, demand_price_blocks=None, techs_methanol=["CO2Compressor", "CO2Storage", "MethanolSynthesis"], co2_label='CO2'):
    """
    Build a restricted Pyomo model for the methanol actor.

    Parameters
    ----------
    cfg : ModelConfig
        Full system configuration, but will be partially overridden.
    co2_supply : dict
        CO2 available supply at each time step.
        Format: {t: quantity, ...}
    demand_price_blocks : dict
        Demand price blocks for each time step, observed from the supplier at the current iteration.
        Format: {t: [{"block": int, "price": float, "capacity": float}, ...], ...}
    techs_methanol : list of str
        Names of technologies to include (default: ["CO2Compressor"]).
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

    print("All data loaded for methanol submodel.")

    # Ensure methanol technologies are in the data
    for tech in techs_methanol:
        if tech not in data['G']:
            techs_methanol.remove(tech)
    
    # # Restrict technologies G
    # data['G'] = [g for g in data['G'] if g in techs_methanol]

    # # Ensure CO2 is in the fuels set
    # if co2_label not in data['F']:
    #     data['F'].append(co2_label)

    # Ensure price_buy has entries for CO2
    area_import = "DK1"  # Assuming import area is DK1; adjust as needed
    price_co2_external = cfg.co2_market_price

    if 'price_buy' not in data:
        data['price_buy'] = {}
    for t in data['T']:
        key = (area_import, co2_label, t)
        data['price_buy'][key] = price_co2_external

    # ------------------------------------------------------------------
    # 2. Assemble Pyomo model
    # ------------------------------------------------------------------
    m = pyo.ConcreteModel()

    m.Demand_Target = cfg.demand_target
    m.GreenElectricity = cfg.green_electricity
    m.ElectricityMandate = cfg.electricity_mandate
    m.ElProdToGrid = cfg.el_prod_to_grid

    # Add a flag to disable generic CO2 balance constraint
    m.SkipCO2Balance = True
    
    define_sets(m, data)
    define_params(m, data, tech_df)
    define_variables(m)
    add_constraints(m)

    # ------------------------------------------------------------------
    # 3. Define CO2 supply (availability constraint): from internal biogas generation or from external market
    # ------------------------------------------------------------------
    # CO2 is available from biogas submodel supply: co2_supply[t] represents the supplied co2 quantity at time t to methanol plant
    # Generation can only be up to the market supplied amount ()
    def co2_supply_limit_rule(m, t):
        return sum(
            m.Generation[g, co2_label, t] 
            for g in m.G
            if (g, co2_label) in m.f_out
            ) <= co2_supply.get(t, 0.0)
    
    m.CO2_SupplyLimit = pyo.Constraint(m.T, rule=co2_supply_limit_rule)

    # The generic balance_rule for CO2 has been disabled in add_constraints()
    # Define CO2 balance constraint specifically for methanol submodel
    def co2_balance_rule(m, t):
        internal = sum(
            m.Generation[g, co2_label, t]
            for g in m.G
            if (g, co2_label) in m.f_out
        )

        external = m.Buy[area_import, co2_label, t]
        
        demand = sum(
            m.Fueluse[g, co2_label, t]
            for g in m.G
            if (g, co2_label) in m.f_in
        )

        return internal + external == demand
    m.CO2_Balance = pyo.Constraint(m.T, rule=co2_balance_rule)

    # Allow to extract internal CO2 use after solving
    def co2_internal_use_rule(m, t):
        return sum(
            m.Generation[g, co2_label, t]
            for g in m.G
            if (g, co2_label) in m.f_out
        )
    m.CO2_InternalUse = pyo.Expression(m.T, rule=co2_internal_use_rule)


    # ------------------------------------------------------------------
    # 4. Modify objective to only consider methanol technologies
    # ------------------------------------------------------------------
    def methanol_objective(m):

        # Reuse the existing "cost" pieces generated by define_objective() but here we rebuild a custom objective.
        # Note: we only include costs and revenues related to the methanol technologies (including the CO2 import costs)
        # - by restricting the summations to the relevant sets and variables.

        fuels_in = set(f for (g,f) in m.f_in if g in techs_methanol)
        fuels_out = set(f for (g,f) in m.f_out if g in techs_methanol)

        # a) Fuel import cost 
        imp_cost = sum(
            m.price_buy[a,e,t] * m.Buy[a,e,t]
            # for (a,e) in m.buyE
            for a in m.A for e in fuels_in if (a,e) in m.buyE
            for t in m.T
        )
        # b) Sale revenue
        sale_rev = sum(
            m.price_sale[a,e,t] * m.Sale[a,e,t]
            for a in m.A for e in fuels_out if (a,e) in m.saleE
            for t in m.T
        )
        # c) Variable O&M on all tech→energy links (only for technologies in biogas submodel)
        var_om = sum(
            m.Generation[g,e,t] * m.cvar[g]
            for g in techs_methanol for e in m.F if (g,e) in m.TechToEnergy
            for t in m.T
        )
        # d) Startup costs (only for technologies in biogas submodel)
        startup = sum(
            m.Startcost[g,t]
            # for g in m.G
            for g in techs_methanol
            for t in m.T
        )
        # e) Slack penalties (both import‐slack and export‐slack)
        DemandSet_restricted = [(a,e,t) for (a,e,t) in m.DemandSet if e in fuels_out]
        DemandFuel_restricted = [(s,f) for (s,f) in m.DemandFuel if f in fuels_out]
        slack_sum = (
            sum(m.SlackDemandImport[a, e, t] + m.SlackDemandExport[a, e, t] for (a, e, t) in DemandSet_restricted)
            + sum(m.SlackTarget[s, f] for (s, f) in DemandFuel_restricted)
        )

        total_profit_expr = sale_rev - imp_cost - var_om - startup - cfg.penalty * slack_sum

        return -total_profit_expr

    m.Obj = pyo.Objective(rule=methanol_objective, sense=pyo.minimize)



    return m
