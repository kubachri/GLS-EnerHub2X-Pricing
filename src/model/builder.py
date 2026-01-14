# src/model/builder.py

from pyomo.environ import ConcreteModel
from src.config import ModelConfig
from src.data.loader import load_data
from src.data.preprocess import scale_tech_parameters, slice_time_series
from src.model.sets import define_sets
from src.model.parameters import define_params
from src.model.variables import define_variables
from src.model.constraints import add_constraints
from src.model.objective import define_objective


def build_model(cfg: ModelConfig) -> ConcreteModel:
    """
    Build and return the Pyomo model based on the provided configuration.

    Parameters
    ----------
    cfg : ModelConfig
        Configuration object controlling test mode, horizon length, and penalty.

    Returns
    -------
    ConcreteModel
        The constructed and fully specified Pyomo optimization model.
    """
    # 1) Load raw data
    data, tech_df = load_data(cfg)

    # 2) Preprocess technology parameters (scaling capacity, minima, ramp rates)
    data, tech_df = scale_tech_parameters(data, tech_df)

    # 3) Optionally slice time-series for test-mode runs
    if cfg.test_mode:
        data = slice_time_series(data, cfg.n_test)

    print("\nAll data loaded.\n")

    # 4) Assemble the model
    model = ConcreteModel()

    #DemandTarget
    model.Demand_Target = cfg.demand_target
    if model.Demand_Target:
        print("Running with a demand target ...\n")

    #Green fuels
    model.GreenElectricity = cfg.green_electricity
    if model.GreenElectricity:
        print("Running with a green electrity from the grid (<20 EUR/MWh) ...\n")

    #Electricity mandate
    model.ElectricityMandate = cfg.electricity_mandate
    if model.ElectricityMandate:
        print(f"Running with an electricity mandate of {model.ElectricityMandate*100}% ...\n")

    #Electricity export limit
    model.ElProdToGrid = cfg.el_prod_to_grid
    if model.ElProdToGrid:
        print(f"Running with grid export to production ratio of {model.ElProdToGrid*100}% ...\n")

    define_sets(model, data)
    define_params(model, data, tech_df)
    define_variables(model)
    add_constraints(model)
    define_objective(model, cfg=cfg)

    return model
