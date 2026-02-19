# src/config.py
from dataclasses import dataclass
import os

DEFAULT_DATA_FILE = os.path.join(os.getcwd(), "scenarios", "Data_Baseline.xlsx")

@dataclass
class ModelConfig:
    test_mode:              bool    = False   # run a short “N_test” horizon?
    n_test:                 int     = 168     # number of hours in test mode
    penalty:                float   = 10**9     # slack‐penalty in objective
    demand_target:          bool    = True
    sensitivity:            bool    = False
    green_electricity:      bool    = False
    electricity_mandate:    float   = 1.0 # it's the ratio of electricity imported/electricity used in EH (limits grid imports)
    el_prod_to_grid:        float   = 1.0 # it's the ratio of electricity exported/electricity produced in EH (limits grid exports)
    data_file:              str     = DEFAULT_DATA_FILE

    @property
    def data_dir(self) -> str:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        return os.path.join(root, self.data)