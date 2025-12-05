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


def build_biogas_model(cfg, co2_price_blocks, techs_biogas=["BiogasUpgrade", "CO2Compressor", "CO2Storage"], co2_label='CO2'):
    """
    Build a restricted Pyomo model for the biogas actor.

    Parameters
    ----------
    cfg : ModelConfig
        Full system configuration, but will be partially overridden.
    co2_price_blocks : list of dicts
        Example format:
            [
                {"block": 1, "price": 40, "capacity": 100},
                {"block": 2, "price": 35, "capacity": 150},
                ...
            ]
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
    # 4. Inject CO2 price-quota blocks
    # ------------------------------------------------------------------
    # Create an indexed set of blocks
    block_ids = [blk["block"] for blk in co2_price_blocks]
    m.CO2_BLOCKS = pyo.Set(initialize=block_ids)

    # Block prices and capacities
    m.CO2_BlockPrice = pyo.Param(
        m.CO2_BLOCKS,
        initialize={blk["block"]: blk["price"] for blk in co2_price_blocks},
        within=pyo.NonNegativeReals,
        mutable=True,
    )

    m.CO2_BlockCap = pyo.Param(
        m.CO2_BLOCKS,
        initialize={blk["block"]: blk["capacity"] for blk in co2_price_blocks},
        within=pyo.NonNegativeReals,
        mutable=True,
    )

    # Decision: how much CO2 is sold in each block
    m.CO2_SellBlock = pyo.Var(m.CO2_BLOCKS, within=pyo.NonNegativeReals)

    # Total CO2 sales from blocks
    def co2_block_sum_rule(mm):
        return sum(mm.CO2_SellBlock[b] for b in mm.CO2_BLOCKS)
    m.CO2_TotalSell = pyo.Expression(rule=co2_block_sum_rule)

    # ------------------------------------------------------------------
    # 5. Link block sales to Sale variable in CO2 fuel balance
    # ------------------------------------------------------------------
    # Simplest version: enforce Sale == block sum for each time and location
    # (Extend later to multiple locations/time if desired)
    # Here assume 1 area and all time steps same price curve.
    first_area = list(data["A"])[0]
    first_t = list(data["T"])[0]

    def bind_sale_to_blocks(mm):
        return mm.Sale[first_area, co2_label, first_t] == mm.CO2_TotalSell
    m.CO2_BlockBinding = pyo.Constraint(rule=bind_sale_to_blocks)

    # ------------------------------------------------------------------
    # 7. Define biogas profit objective
    # ------------------------------------------------------------------
    def biogas_profit_rule(mm):
        revenue = sum(
            mm.CO2_BlockPrice[b] * mm.CO2_SellBlock[b]
            for b in mm.CO2_BLOCKS
        )
        # Use internal cost expression if needed (var_om, electricity cost...)
        # Reuse the existing "cost" pieces generated by define_objective()
        # but here we rebuild a custom objective.

        # Compute costs: variable O&M and electricity purchases
        # You may want to refine this later to match your exact biogas cost formulation.
        base_cost = 0
        if hasattr(mm, "Total_VarOM_Cost"):
            base_cost += mm.Total_VarOM_Cost
        if hasattr(mm, "Total_Imp_Cost"):
            base_cost += mm.Total_Imp_Cost

        return revenue - base_cost

    m.Obj = pyo.Objective(rule=biogas_profit_rule, sense=pyo.maximize)

    return m
