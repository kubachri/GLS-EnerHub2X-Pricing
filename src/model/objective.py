# src/model/objective.py

from pyomo.environ import Constraint, Objective, maximize, value
from src.config import ModelConfig

def define_objective(m, cfg: ModelConfig):
    # ProfitDefinition handled in a constraint
    penalty = cfg.penalty

    # a) Fuel cost (imports are a positive cost → negative in objective)
    imp_cost = sum(
        m.price_buy[a,e,t] * m.Buy[a,e,t]
        for (a,e) in m.buyE
        for t in m.T
    )
    # b) Sale revenue
    sale_rev = sum(
        m.price_sale[a,e,t] * m.Sale[a,e,t]
        for (a,e) in m.saleE
        for t in m.T
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

    total_profit_expr = sale_rev -imp_cost - var_om - startup - penalty * slack_sum

    m.Obj = Objective(expr=total_profit_expr, sense=maximize)

def debug_objective(m, cfg):
    # 1) Recompute each piece
    imp_cost = sum(value(m.price_buy[a,e,t] * m.Buy[a,e,t])
                   for (a,e) in m.buyE for t in m.T)
    sale_rev = sum(value(m.price_sale[a,e,t] * m.Sale[a,e,t])
                   for (a,e) in m.saleE for t in m.T)
    var_om   = sum(value(m.Generation[g,e,t] * m.cvar[g])
                   for (g,e) in m.TechToEnergy for t in m.T)
    startup  = sum(value(m.Startcost[g,t])
                   for g in m.G for t in m.T)

    # 2) Sum *all* slack, import + export
    slack_imp = sum(value(v) for v in m.SlackDemandImport.values())
    slack_exp = sum(value(v) for v in m.SlackDemandExport.values())
    slack_sum = slack_imp + slack_exp
    slack_pen = cfg.penalty * slack_sum

    # 3) “Hand” objective
    manual_obj = imp_cost - sale_rev + var_om + startup + slack_pen

    # 4) Pyomo’s objective
    pyomo_obj = value(m.Obj.expr)  # or value(m.Profit)

    # 5) Print breakdown
    print("\n=== OBJECTIVE BREAKDOWN ===")
    print(f"Import Costs     : {imp_cost:,.2f}")
    print(f"Sales Revenue    : {sale_rev:,.2f}")
    print(f"Variable O&M      : {var_om:,.2f}")
    print(f"Startup Costs    : {startup:,.2f}")
    print(f"Slack Import sum : {slack_imp:,.2f}")
    print(f"Slack Export sum : {slack_exp:,.2f}")
    print(f"Penalty factor   : {cfg.penalty:e}")
    print(f"Slack penalty    : {slack_pen:,.2f}")
    print(f"―" * 40)
    print(f"Manual objective : {manual_obj:,.2f}")
    print(f"Pyomo objective  : {pyomo_obj:,.2f}")
    print("  (difference  = "
          f"{(pyomo_obj - manual_obj):.6f})\n")

    # 6) List any nonzero slack variables
    print("Nonzero Slack variables:")
    for idx, var in m.SlackDemandImport.items():
        val = value(var)
        if val > 1e-8:
            print(f"  Import {idx} = {val:,.2f}")
    for idx, var in m.SlackDemandExport.items():
        val = value(var)
        if val > 1e-8:
            print(f"  Export {idx} = {val:,.2f}")
    print("============================\n")