# src/model/sets.py

from pyomo.environ import Set
from collections import defaultdict

def define_sets(model, data):
    """
    Define Pyomo Sets on the model using preprocessed data:
      - A: areas
      - G: all technologies
      - F: fuels/energies
      - T: time periods (possibly sliced in test mode)
      - G_s: storage-only technologies
      - G_p: production-only technologies
      - UC: unit-commitment technologies
      - RR: ramp-rate technologies
      - flowset: inter-area flow links
      - TechToEnergy: mapping of tech→energy exports
      - f_in, f_out: tech-energy import/export pairs
      - buyE, saleE: market buy/sell interfaces
      - location: (area, tech) location pairs
      - LinesInterconnectors: areas with any interconnector capacity
    """

    # Core entity sets
    G_p = [g for g in data['G'] if g not in data['G_s']]

    # Demand
    raw_demand = data['Demand']

    # Fuel import/export pairs
    pairs_out = [(g, f) for (g, f), out in data['sigma_out'].items() if out > 0]
    pairs_in  = [(g, f) for (g, f), inp in data['sigma_in'].items()  if inp > 0]

    # Market interfaces (areas × fuels with positive prices)
    buy_pairs  = sorted({(a, e) for (a, e, t), p in data['price_buy'].items()  if p > 0})
    sale_pairs = sorted({(a, e) for (a, e, t), p in data['price_sell'].items() if p > 0})

    #Designated technology - fuel pairs
    tech_to_f = [(g,f) for (g,f), out in data['sigma_out'].items() if out == 1]

    # Areas that have interconnector capacity
    area_has = defaultdict(bool)
    for (a, f, t), cap in data['Xcap'].items():
        if cap > 0:
            area_has[a] = True
    lines = [a for a, has in area_has.items() if has]

    # Sets definition
    model.A = Set(initialize=data['A'])
    model.G = Set(initialize=data['G'], ordered=True)
    model.F = Set(initialize=data['F'])
    model.T = Set(initialize=data['T'], ordered=True)

    # Storage vs production technologies
    model.G_s = Set(initialize=data['G_s'], within=model.G)
    model.G_p = Set(initialize=G_p, within=model.G)

    # Technology location (area, tech)
    model.location = Set(
        initialize=data['location'],
        dimen=2,
        within=model.A * model.G
    )

    # Unit-commitment and ramp-rate
    model.UC = Set(initialize=data['UC'], within=model.G)
    # Inter-area flow definitions
    model.flowset = Set(
        initialize=data['FlowSet'],
        dimen=3,
        within=model.A * model.A * model.F
    )

    model.f_out = Set(initialize=pairs_out, dimen=2, within=model.G * model.F)
    model.f_in  = Set(initialize=pairs_in,  dimen=2, within=model.G * model.F)
    model.buyE  = Set(initialize=buy_pairs,  dimen=2, within=model.A * model.F)
    model.saleE = Set(initialize=sale_pairs, dimen=2, within=model.A * model.F)
    model.LinesInterconnectors = Set(initialize=lines, within=model.A)
    model.DemandSet = Set(initialize=raw_demand.keys(), dimen=3, within=model.A * model.F * model.T)
    model.TechToEnergy = Set(initialize=tech_to_f, dimen=2, within=model.G * model.F)

    demand_target_keys = data['DemandTarget'].keys()
    model.DemandFuel = Set(dimen=2, initialize=demand_target_keys)
    model.DemandSteps = Set(initialize=sorted({step for (step, _) in demand_target_keys}))
    model.Weeks = Set(initialize=sorted(set(data['WeekOfT'].values())))