# src/constraints.py

from pyomo.environ import Constraint

# 1) Flows imported to technologies
def fuelmix_rule(m, g, e, t):
    if m.capacity[g] <= 0:
        print(f'Technology {g} does not have a capacity value.')
        return Constraint.Skip
    return m.in_frac[g,e] * m.Fuelusetotal[g,t] == m.Fueluse[g,e,t]

# 2) Production for each non-storage technology
def production_rule(m, g, e, t):
    if m.capacity[g] <= 0:
        print(f'Technology {g} does not have a capacity value.')
        return Constraint.Skip
    if g in m.G_s:
        return Constraint.Skip
    return m.out_frac[g,e] * m.Fuelusetotal[g,t] * m.Fe[g] == m.Generation[g,e,t]

# 3) Storage constraints
def storage_balance_rule(m, g, t):
    if g not in m.G_s:
        return Constraint.Skip
    prev = m.soc_init[g] if t==m.T.first() else m.Volume[g,m.T.prev(t)]
    discharge = sum(
        m.Generation[g,e,t] * m.out_frac[g,e]
        for (gg,e) in m.f_out
        if gg==g)
    return m.Volume[g,t] == prev + m.Fuelusetotal[g,t] * m.Fe[g] - discharge

def charging_max(m, g, t):
    if g not in m.G_s:
        return Constraint.Skip
    return m.Fuelusetotal[g,t] <= m.capacity[g] * m.Charge[g,t]

def discharging_max(m, g, t):
    if g not in m.G_s:
        return Constraint.Skip
    discharge = sum(
        m.Generation[g,e,t] * m.out_frac[g,e]
        for (gg,e) in m.f_out
        if gg==g)
    return discharge <= m.capacity[g] * (1-m.Charge[g,t])

# def charging_min(m, g, t):
#     if g not in m.G_s or m.Minimum[g] <=0 :
#         return Constraint.Skip
#     return m.Fuelusetotal[g,t] <= m.Minimum[g] * m.Charge[g,t]

#
# def discharging_min(m, g, t):
#     if g not in m.G_s or m.Minimum[g] <=0:
#         return Constraint.Skip
#     discharge = sum(
#         m.Generation[g,e,t] * m.out_frac[g,e]
#         for (gg,e) in m.f_out
#         if gg==g)
#     return discharge <= m.Minimum[g] * (1-m.Charge[g,t])

def volume_upper_rule(m, g, t):
    return m.Volume[g, t] <= m.soc_max[g]


def volume_final_soc(m, g):
    return m.Volume[g, m.T.last()] == m.soc_init[g]

# 4) Energy balance equations
def balance_rule(m, a, e, t):
    # 1) GAMS $‐guard: buyE(a,e) OR saleE(a,e) OR any tech at area a with (in or out) of e
    has_buy  = (a,e) in m.buyE
    has_sale = (a,e) in m.saleE
    # find all techs located in area a
    techs_in_area = [tech for (area,tech) in m.location if area==a]
    has_tech = any(
        ((tech,e) in m.f_in) or ((tech,e) in m.f_out)
        for tech in techs_in_area
    )
    if not (has_buy or has_sale or has_tech):
        return Constraint.Skip
    # 2) Left‐hand side: Buy + inbound flows + local generation
    buy_term = m.Buy[a,e,t] if has_buy else 0.0
    inflow   = sum(
        m.Flow[area_in, a, e, t]
        for (area_in, area_to, energy) in m.flowset
        if (area_to==a and energy==e)
    )
    generation = sum(
        m.Generation[tech, e, t]
        for tech in techs_in_area
        if (tech,e) in m.f_out
    )
    # 3) Right‐hand side: local fuel use + Sale + outbound flows
    fueluse   = sum(
        m.Fueluse[tech, e, t]
        for tech in techs_in_area
        if (tech,e) in m.f_in
    )
    sale_term = m.Sale[a,e,t] if has_sale else 0.0
    outflow   = sum(
        m.Flow[a, area_out, e, t]
        for (area_from, area_out, energy) in m.flowset
        if (area_from==a and energy==e)
    )
    # 4) Assemble the balance
    return buy_term + inflow + generation == fueluse + sale_term + outflow

# # # 5) Demand constraint based on sales + slack
# def demand_time_rule(m, a, e, t):
#     # 1) GAMS $-guard: only if there is any demand at all for (a,e)
#     total_area_energy = sum(m.demand[a,e,tt] for tt in m.T)
#     if total_area_energy <= 0:
#         return Constraint.Skip
#     # 2) LHS terms, zero when not defined in the corresponding set
#     sale_term = m.Sale[a,e,t] if (a,e) in m.saleE else 0.0
#     # slack_imp = m.SlackDemandImport[a,e,t] if (a,e) in m.buyE  else 0.0
#     # slack_exp = m.SlackDemandExport[a,e,t] if (a,e) in m.saleE else 0.0
#     slack_imp = m.SlackDemandImport[a,e,t]
#     slack_exp = m.SlackDemandExport[a,e,t]
#     lhs = sale_term + slack_imp - slack_exp
#     # 3) RHS is the exact demand
#     rhs = m.demand[a,e,t]
#     return lhs >= rhs

# 5) Demand constraint based on generation + slack
def demand_time_rule(m, a, e, t):
    # Skip if no demand
    if m.demand[a,e,t] == 0:
        return Constraint.Skip

    # 1) Local generation of e in area a
    local_gen = sum(
        m.Generation[tech, e, t]
        for (area, tech) in m.location
        if area == a and (tech, e) in m.f_out
    )

    # 2) Inflow from other areas
    inflow = sum(
        m.Flow[area_from, a, e, t]
        for (area_from, area_to, fuel) in m.flowset
        if area_to == a and fuel == e
    )

    # 3) Slack for unmet demand
    slack_imp = m.SlackDemandImport[a,e,t]

    # 4) Total supply ≥ demand
    return (local_gen + inflow + slack_imp) >= m.demand[a,e,t]


# 6) MaxBuy
def max_buy_rule(m, e, t):
    # total capacity for e,t across your interconnector‐areas
    total_cap = sum(
        m.InterconnectorCapacity[a,e,t]
        for a in m.LinesInterconnectors
        if (a,e,t) in m.InterconnectorCapacity.index_set()
    )
    if total_cap <= 0:
        # no lines for this energy/time → skip
        return Constraint.Skip
    # sum of all buys for this energy/time
    lhs = sum(
        m.Buy[a,e,t]
        for (a,f) in m.buyE
        if f == e
    )
    # average capacity per area:
    rhs = total_cap / len(m.LinesInterconnectors)
    return lhs <= rhs

# 7) MaxSale (analogous)
def max_sale_rule(m, e, t):
    total_cap = sum(
        m.InterconnectorCapacity[a,e,t]
        for a in m.LinesInterconnectors
        if (a,e,t) in m.InterconnectorCapacity.index_set()
    )
    if total_cap <= 0:
        return Constraint.Skip
    lhs = sum(
        m.Sale[a,e,t]
        for (a,f) in m.saleE
        if f == e
    )
    rhs = total_cap / len(m.LinesInterconnectors)
    return lhs <= rhs

def availability_rule(m, g, t):
    # skip storage, skip zero‐cap techs
    if g in m.G_s or m.capacity[g] <= 0:
        return Constraint.Skip
    # total fuel‐use (pre‐efficiency) cannot exceed capacity×profile
    return m.Fuelusetotal[g, t] <= m.capacity[g] * m.Profile[g, t]

# 8) Ramp-up: (Generation[t] – Generation[t-1])/Fe ≤ LHS
def ramp_up_rule(m, g, t):
    # only where a nonzero RampRate exists, and skip the first hour
    if g not in m.UC or t == m.T.first():
        return Constraint.Skip
    prev_on = m.Online[g, m.T.prev(t)]
    lhs = m.RampRate[g]*prev_on + m.Minimum[g]*(1-prev_on)
    # right‐hand side: sum over export‐energies of (Gen[t]–Gen[t-1])/Fe
    rhs = sum(
        (m.Generation[g,e,t] - m.Generation[g,e,m.T.prev(t)])/m.Fe[g]
        for (gg,e) in m.f_out
        if gg == g
    )
    return lhs >= rhs

# 9) Ramp-down: (Generation[t-1] – Generation[t])/Fe ≤ LHS
def ramp_down_rule(m, g, t):
    if g not in m.UC or t == m.T.first():
        return Constraint.Skip
    lhs = m.RampRate[g]*m.Online[g,t] + m.Minimum[g]*(1-m.Online[g,t])
    # RHS: sum over export‐energies of (Gen[t-1]–Gen[t])/Fe
    rhs = sum(
        (m.Generation[g,e,m.T.prev(t)] - m.Generation[g,e,t]) / m.Fe[g]
        for (gg,e) in m.f_out
        if gg == g
    )
    return lhs >= rhs

# 10) Capacity constraint: Capacity*Online ≥ FuelUseTotal  (only for UC)
def capacity_rule(m, g, t):
    if g not in m.UC:
        return Constraint.Skip
    return m.capacity[g] * m.Online[g,t] >= m.Fuelusetotal[g,t]

# 11) Minimum-load: FuelUseTotal ≥ Minimum*Online  (only if Minimum>0)
def minimum_load_rule(m, g, t):
    if g not in m.UC or m.Minimum[g] <= 0:
        return Constraint.Skip
    return m.Fuelusetotal[g,t] >= m.Minimum[g] * m.Online[g,t]

# 12) Startup cost: Startcost ≥ StartupCost*(Online[t]–Online[t-1])  (only if cstart>0)
def startup_cost_rule(m, g, t):
    if g not in m.UC or m.cstart[g] <= 0:
        return Constraint.Skip
    # treat previous‐hour Offline before t=first as 0
    prev_on = 0 if t == m.T.first() else m.Online[g, m.T.prev(t)]
    return m.Startcost[g,t] >= m.cstart[g] * (m.Online[g,t] - prev_on)

# 13) Electricity Mandate (Green H2)
def green_electricity_import(m, a, e, t):
    if (a, e) != ('DK1', 'Electricity'):
        return Constraint.Skip
    if m.price_buy[a, e, t] > 20.0:
        return m.Buy[a, e, t] == 0
    return Constraint.Skip
    
def restrict_grid_import(m, t):
    # Grid electricity buy at time t — only DK1
    grid_buy = m.Buy['DK1', 'Electricity', t]

    # Total electricity used by all technologies at time t
    total_electricity_use = sum(
        m.Fueluse[g, 'Electricity', t]
        for (g, f) in m.f_in
        if f == 'Electricity'
    )

    return grid_buy <= m.ElectricityMandate* total_electricity_use

def restrict_grid_export(m, t):
    # Grid electricity sale at time t — only DK1
    grid_sale = m.Sale['DK1', 'Electricity', t]

    # Total electricity produced at time t by any tech that exports electricity
    total_generation = sum(
        m.Generation[g, 'Electricity', t]
        for (g, f) in m.f_out
        if f == 'Electricity'
    )

    return grid_sale <= m.ElProdToGrid * total_generation

# 14) Weekly demand
def target_demand_rule(m, step, area_fuel):
    area, fuel = area_fuel.split('.')
    total = sum(
        m.Generation[g, fuel, t]
        for g in m.G
        for t in m.T
        if (g, fuel) in m.f_out and m.WeekOfT[t] == step
    )
    return total + m.SlackTarget[step, area_fuel] >= m.DemandTarget[step, area_fuel]

def add_constraints(model):
    model.Fuelmix = Constraint(model.f_in, model.T, rule=fuelmix_rule)
    model.Production = Constraint(model.f_out, model.T, rule=production_rule)
    model.ProductionStorage = Constraint(model.G_s, model.T, rule=storage_balance_rule)
    model.ChargingStorageMax = Constraint(model.G_s, model.T, rule=charging_max)
    model.DisChargingStorageMax = Constraint(model.G_s, model.T, rule=discharging_max)
    # model.ChargingStorageMin = Constraint(model.G_s, model.T, rule=charging_min)
    # model.DisChargingStorageMin = Constraint(model.G_s, model.T, rule=discharging_min)
    model.VolumeUpper = Constraint(model.G_s, model.T, rule=volume_upper_rule)
    model.TerminalSOC = Constraint(model.G_s, rule=volume_final_soc)
    model.Balance = Constraint(model.A, model.F, model.T, rule=balance_rule)
    model.DemandTime = Constraint(model.DemandSet, rule=demand_time_rule)
    model.MaxBuy = Constraint(model.F, model.T, rule=max_buy_rule)
    model.MaxSale = Constraint(model.F, model.T, rule=max_sale_rule)
    model.Availability = Constraint(model.G_p, model.T, rule=availability_rule)
    model.RampUp = Constraint(model.G, model.T, rule=ramp_up_rule)
    model.RampDown = Constraint(model.G, model.T, rule=ramp_down_rule)
    model.Capacity = Constraint(model.G, model.T, rule=capacity_rule)
    model.MinimumLoad = Constraint(model.G, model.T, rule=minimum_load_rule)
    model.StartupCost = Constraint(model.G, model.T, rule=startup_cost_rule)
    model.TargetDemand = Constraint(model.DemandFuel, rule=target_demand_rule)
    if model.GreenElectricity:
        model.GreenGrid = Constraint(model.buyE, model.T, rule=green_electricity_import)
    if model.ElectricityMandate:
        model.GridRestriction = Constraint(model.T, rule=restrict_grid_import)
    if model.ElProdToGrid:
        model.ExportLimit = Constraint(model.T, rule=restrict_grid_export)