# src/model/variables.py

from pyomo.environ import Var, NonNegativeReals, Binary, Reals

def define_variables(model):
    """
    Declare all Pyomo Variables for the model:
      - Fueluse, Fuelusetotal, Generation, Volume, Flow
      - Buy, Sale, SlackDemandImport, SlackDemandExport
      - Startcost, Online, Charge
      - Profit
    """
    model.Fueluse           = Var(model.f_in,       model.T, domain=NonNegativeReals)
    model.Fuelusetotal      = Var(model.G,          model.T, domain=NonNegativeReals)
    model.Generation        = Var(model.f_out,      model.T, domain=NonNegativeReals)
    model.Volume            = Var(model.G_s,        model.T, domain=NonNegativeReals)
    model.Flow              = Var(model.flowset,    model.T, domain=NonNegativeReals)
    model.Buy               = Var(model.buyE,       model.T, domain=NonNegativeReals)
    model.Sale              = Var(model.saleE,      model.T, domain=NonNegativeReals)
    model.SlackDemandImport = Var(model.DemandSet,           domain=NonNegativeReals)
    model.SlackDemandExport = Var(model.DemandSet,           domain=NonNegativeReals)
    model.Startcost         = Var(model.G,          model.T, domain=NonNegativeReals)
    model.Online            = Var(model.G,          model.T, domain=Binary)
    model.Charge            = Var(model.G_s,        model.T, domain=Binary)
    model.SlackTarget       = Var(model.DemandFuel,          domain=NonNegativeReals)