# src/strategic/submodel_biogas.py

import pyomo.environ as pyo
from copy import deepcopy

from src.config import ModelConfig
from src.data.loader import load_data   # reused but we will override some parts
from src.data.preprocess import scale_tech_parameters, slice_time_series
from src.model.sets import define_sets
from src.model.parameters import define_params
from src.model.variables import define_variables
from src.model.constraints import add_constraints
from src.model.objective import define_objective
from src.utils.validate_strategic_model import validate_strategic_model


def build_biogas_model(cfg, demand_price_blocks, techs_biogas=["BiogasUpgrade", "CO2Compressor", "CO2Storage"], co2_label='CO2Comp'):
    """
    Build a restricted Pyomo model for the biogas actor.

    Parameters
    ----------
    cfg : ModelConfig
        Full system configuration, but will be partially overridden.
    demand_price_blocks : dict with time t as key, value list of dicts (ordered decreasingly by price)
        Example format:
            {
                1: [
                    {"block": 1, "price": 50, "capacity": 200},
                    {"block": 2, "price": 45, "capacity": 150},
                    ...
                ],
                2: [...],
                ...
            }
    techs_biogas : list of str
        Names of technologies to include.
    co2_label : str
        Fuel name used in the model.

    Returns
    -------
    model : ConcreteModel
    """

    # ------------------------------------------------------------------
    # 1. Load full data and restrict to biogas technologies only
    # ------------------------------------------------------------------
    data, tech_df = load_data(cfg)

    data = deepcopy(data)  # do not mutate original session memory
    tech_df = deepcopy(tech_df)

    data, tech_df = scale_tech_parameters(data, tech_df)

    if cfg.test_mode:
        data = slice_time_series(data, cfg.n_test)

    print("\nAll data loaded for biogas submodel.\n")
    
    # Restrict technologies G
    data['G'] = [g for g in data['G'] if g in techs_biogas]

    # Ensure CO2 is in the fuels set for Sale variable definition
    if co2_label not in data['F']:
        data['F'].append(co2_label)

    # Ensure price_sell has entries for CO2 (set to 0, since price is endogenous)
    if 'price_sell' not in data:
        data['price_sell'] = {}
    for a in data['A']:
        for t in data['T']:
            key = (a, co2_label, t)
            if key not in data['price_sell']:
                data['price_sell'][key] = 0.0

    # Optionally restrict other sets if needed (Locations, Fuels)
    # Here we keep full sets but it might be better to restrict further.

    # ------------------------------------------------------------------
    # 2. Assemble Pyomo model
    # ------------------------------------------------------------------
    m = pyo.ConcreteModel()

    # #DemandTarget
    # m.Demand_Target = cfg.demand_target
    # if m.Demand_Target:
    #     print("Running with a demand target ...\n")

    # #Green fuels
    # m.GreenElectricity = cfg.green_electricity
    # if m.GreenElectricity:
    #     print("Running with a green electrity from the grid (<20 EUR/MWh) ...\n")

    # #Electricity mandate
    # m.ElectricityMandate = cfg.electricity_mandate
    # if m.ElectricityMandate:
    #     print(f"Running with an electricity mandate of {m.ElectricityMandate*100}% ...\n")

    # #Electricity export limit
    # m.ElProdToGrid = cfg.el_prod_to_grid
    # if m.ElProdToGrid:
    #     print(f"Running with grid export to production ratio of {m.ElProdToGrid*100}% ...\n")

    define_sets(m, data)
    define_params(m, data, tech_df)
    define_variables(m)
    add_constraints(m)
    
    # define_objective(m, cfg=cfg)

    # ------------------------------------------------------------------
    # 3. Inject CO2 demand price curve (time-dependent)
    # ------------------------------------------------------------------
    # Create indexed sets of blocks per time
    def demand_blocks_init(m, t):
        return [blk["block"] for blk in demand_price_blocks.get(t, [])]
    m.DEMAND_BLOCKS = pyo.Set(m.T, initialize=demand_blocks_init)

    # Block prices per time and block
    def demand_block_price_init(m, t, b):
        blocks = demand_price_blocks.get(t, [])
        return next((blk["price"] for blk in blocks if blk["block"] == b), 0)
    m.Demand_BlockPrice = pyo.Param(
        m.T, m.DEMAND_BLOCKS,
        initialize=demand_block_price_init,
        within=pyo.NonNegativeReals,
        mutable=True,
    )

    # Block capacities per time and block
    def demand_block_cap_init(m, t, b):
        blocks = demand_price_blocks.get(t, [])
        return next((blk["capacity"] for blk in blocks if blk["block"] == b), 0)
    m.Demand_BlockCap = pyo.Param(
        m.T, m.DEMAND_BLOCKS,
        initialize=demand_block_cap_init,
        within=pyo.NonNegativeReals,
        mutable=True,
    )

    # Cumulative capacities per time and block
    def demand_block_cumcap_init(m, t, b):
        blocks = demand_price_blocks.get(t, [])
        return sum(blk["capacity"] for blk in blocks if blk["block"] < b)
    m.Demand_BlockCumCap = pyo.Param(
        m.T, m.DEMAND_BLOCKS,
        initialize=demand_block_cumcap_init,
        within=pyo.NonNegativeReals,
        mutable=True,
    )    

    # ------------------------------------------------------------------
    # 4. Define variables for CO2 block sales (time-dependent)
    # ------------------------------------------------------------------

    # Decision: in which block is the market cleared per time
    m.CO2_ActiveBlock = pyo.Var(m.T, m.DEMAND_BLOCKS, within=pyo.Binary)

    # Decision: how much CO2 is sold in the active block per time
    m.CO2_SellBlock = pyo.Var(m.T, m.DEMAND_BLOCKS, within=pyo.NonNegativeReals)

    # Total CO2 sales from blocks per time
    def co2_total_sell_rule(m, t):
        return sum(m.CO2_SellBlock[t, b] + m.CO2_ActiveBlock[t, b] * m.Demand_BlockCumCap[t, b] for b in m.DEMAND_BLOCKS[t])
    m.CO2_TotalSell = pyo.Expression(m.T, rule=co2_total_sell_rule)

    # CO2 market clearing price expression per time
    def co2_market_price_rule(m, t):
        return sum(m.Demand_BlockPrice[t, b] * m.CO2_ActiveBlock[t, b] for b in m.DEMAND_BLOCKS[t])
    m.CO2_MarketPrice = pyo.Expression(m.T, rule=co2_market_price_rule)

    # ------------------------------------------------------------------
    # 5. Define constraints for CO2 block sales (time-dependent)
    # ------------------------------------------------------------------

    def co2_block_capacity_rule(m, t, b):
        return m.CO2_SellBlock[t, b] <= m.Demand_BlockCap[t, b] * m.CO2_ActiveBlock[t, b]
    m.CO2_BlockCapConstr = pyo.Constraint(m.T, m.DEMAND_BLOCKS, rule=co2_block_capacity_rule)

    def co2_single_active_block_rule(m, t):
        return sum(m.CO2_ActiveBlock[t, b] for b in m.DEMAND_BLOCKS[t]) == 1
    m.CO2_SingleActiveBlockConstr = pyo.Constraint(m.T, rule=co2_single_active_block_rule)

    # ------------------------------------------------------------------
    # 6. Link block sales to Sale variable in CO2 fuel balance
    # ------------------------------------------------------------------
    # Simplest version: enforce Sale == block sum for each time and location
    # (Extend later to multiple locations/time if desired)

    area_export = "DK1"

    def bind_sale_to_blocks(m, t):
        return m.Sale[area_export, co2_label, t] == m.CO2_TotalSell[t]
    m.CO2_BlockBinding = pyo.Constraint(m.T, rule=bind_sale_to_blocks)


    # ------------------------------------------------------------------
    # 7. Define biogas profit objective
    # ------------------------------------------------------------------
    def biogas_profit_rule(m):

        # Reuse the existing "cost" pieces generated by define_objective()
        # but here we rebuild a custom objective.

        penalty = cfg.penalty

        # a) Fuel cost 
        imp_cost = sum(
            m.price_buy[a,e,t] * m.Buy[a,e,t]
            for (a,e) in m.buyE
            for t in m.T
        )
        # b) Sale revenue (modified to match block pricing, summed over time)
        sale_rev = sum(
            m.Demand_BlockPrice[t, b] * (m.CO2_SellBlock[t, b] + m.CO2_ActiveBlock[t, b] * m.Demand_BlockCumCap[t, b])
            for t in m.T
            for b in m.DEMAND_BLOCKS[t]
        )

        # c) Variable O&M on all tech→energy links
        var_om = sum(
            m.Generation[g,e,t] * m.cvar[g]
            for (g,e) in m.TechToEnergy
            for t in m.T
        )
        # d) Startup costs
        startup = sum(
            m.Startcost[g,t]
            for g in m.G
            for t in m.T
        )
        # e) Slack penalties (both import‐slack and export‐slack)
        slack_sum = (
            sum(m.SlackDemandImport[a, e, t] + m.SlackDemandExport[a, e, t] for (a, e, t) in m.DemandSet)
            + sum(m.SlackTarget[s, f] for (s, f) in m.DemandFuel)
        )

        total_profit_expr = sale_rev - imp_cost - var_om - startup - penalty * slack_sum

        return total_profit_expr

    m.Obj = pyo.Objective(rule=biogas_profit_rule, sense=pyo.maximize)



    return m
