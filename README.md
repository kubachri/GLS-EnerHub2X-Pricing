# EnerHub2X-Pricing

EnerHub2X-Pricing is a Python/Pyomo implementation of a MILP optimisation model for system analysis of integrated energy hubs. The model **optimises hourly operation** of the hub across multiple energy carriers, including electricity, hydrogen, CO₂, and synthetic fuels.

The original formulation of this model was developed by **Ioannis Kountouris** in **GAMS** for the analysis of energy hubs and Power-to-X systems. EnerHub2X-Pricing is an implementation in Python, extended to include fair-price mechanisms for CO₂ based on shadow prices.

The model is solved using the **Gurobi** optimiser and uses Excel-based inputs to ensure transparency, traceability, and reproducibility.

This repository includes:
- A **Python/Pyomo re-implementation** of the original GAMS model
- Structural refactoring for modularity and maintainability
- Extended diagnostics and export functionality
- A dedicated emphasis on carbon pricing and marginal value analysis

The mathematical structure and modelling philosophy closely follow the original GAMS formulation, while extending it to support pricing‑oriented studies and modern Python workflows.

## What can the model do?

The model solves a profit-maximisation problem to **determine optimal production, storage, conversion, and trading decisions** for an integrated energy hub under technical, economic, and policy constraints.

Specifically, the model can:

- Maximize total system profit
- Optimize hourly dispatch across multiple energy carriers (electricity, hydrogen, CO₂, methanol, heat, etc.)
- Respect technical and operational constraints such as capacity, ramping, and storage limits
- Represent unit commitment and startup costs
- Model storage dynamics with explicit charge/discharge decisions
- Enforce demand targets and policy‑driven constraints
- Compute shadow prices for multiple energy constraints
- Diagnose infeasibilities using automatic IIS extraction

## Model Scope and Assumptions

- Hourly time resolution (typically one full year)
- Fixed or exogenously defined capacities (no endogenous expansion by default)
- Centralised optimisation of a single energy hub
- Perfect foresight over the modelled time horizon
- Linear and mixed‑integer linear constraints
- Internal energy and material balances fully enforced

## Energy Carriers and Technologies

### Energy and Material Carriers
- Electricity
- Hydrogen
- CO₂
- Methanol
- Heat
- Biogas
- Biomethane

### Conversion Technologies
- Electrolysis (hydrogen production)
- CO₂ hydrogenation (e‑methanol)
- Biogas
- Biogas Upgrade (CO₂ source)
- Compression and conversion units

### Storage Technologies
- Electrical storage (battery)
- Hydrogen storage
- CO₂ storage

Technologies and carrier links are defined explicitly via Excel input data (Carriermix).

## External Markets and Carbon Pricing

The hub behaves as a **price taker** with respect to external markets. Prices and availability of external resources are exogenously defined.

External market inputs include:
- Electricity import and export prices (Day-ahead market)
- Natural gas seasonal prices
- E-fuel selling prices
- Interconnector capacities
- Policy‑driven price filters (e.g. green electricity constraints)

Carbon pricing enters the optimisation directly through:
- CO₂ balance constraints
- Shadow prices on CO₂‑related constraints

## Workflow

1. **Input preparation**  
   All model inputs are provided via a single Excel file, including technologies, prices, demands, and time series. By default, the model uses the baseline scenario located in:
   scenarios/Data_Baseline.xlsx
   
   This default is defined in `config.py` as:
   DEFAULT_DATA_FILE = os.path.join(os.getcwd(), "scenarios", "Data_Baseline.xlsx")

3. **Model build**  
   The Pyomo model is assembled modularly: sets → parameters → variables → constraints → objective.

4. **MIP solve**  
   The mixed‑integer problem is solved using Gurobi.

5. **LP re‑solve for pricing**  
   Integer variables are fixed, and the model is re‑solved as a linear program to extract dual values.

6. **Export and diagnostics**  
   Result and input files are exported to the "results" folder in Excel format.

## Model Architecture

The model follows a modular structure:

- config.py: Global model configuration.
- data/: Excel data loading, preprocessing, and sensitivity cases
- model/: Sets, parameters, variables, constraints, and objective
- utils/: Export tools, diagnostics, and infeasibility analysis
- model_run.py: Execusion script

## Software Requirements

- Python 3.10 or newer
- Gurobi Optimiser with a valid license (Any open-source solver can also work)
- Python dependencies listed in requirements.txt

## Installation

Install Python dependencies using:

pip install -r requirements.txt

## Running the model (via terminal)

### Single-scenario run (default)
Running the model without specifying a data file executes the default baseline scenario:
python model_run.py

### Multiple-scenario run
The model can be run in multi-scenario mode by iterating over all Excel files found in the `scenarios/` directory.
python model_run.py --multiple_scenarios true
In this mode, each Excel file in `scenarios/` is treated as an independent scenario and solved sequentially.

Optional flags allow:
- Short horizon test runs
- Penalty tuning for slack variables
- Enabling or disabling demand targets
- Electricity import/export mandates
- Sensitivity case execution
- Running multiple scenarios by iterating over all Excel files in the `scenarios/` folder

## Input Data

All model inputs are provided via a single Excel workbook, including:
- Technology definitions and locations
- Carrier mixes and efficiencies
- Capacities, ramp rates, and startup costs
- Hourly demand profiles
- Import and export prices
- Interconnector capacities

## Output Data

After a successful run, the model exports:
- Hourly operational results
- Aggregated technology and area summaries
- Energy flows between areas
- Market transactions
- Dual values (shadow prices) for selected constraints
- A full export of the inputs considered in the run

Outputs are written to the results directory in Excel format.

## Infeasibility Handling

If the model is infeasible, the framework can:
- Compute an Irreducible Infeasible Set (IIS)
- Export the IIS for inspection in Gurobi
- Report the largest constraint violations

## Applications

- Power-to-X system analysis
- Industrial energy hubs
- CO₂ capture, utilization, and storage systems
- Energy market participation studies
- Carbon pricing and policy impact assessment
