from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    mode_avail: str
    objective: str
    phase_b_upper_bound: int


@dataclass
class SolveOutcome:
    quantities: Dict[str, int]
    objective_value: float
    status: str
    solver_used: str
    fallback_used: bool
    method: str
    elapsed_time_sec: float


@dataclass
class RunOutputs:
    fg_result: pd.DataFrame
    rm_diagnostic: pd.DataFrame
    run_meta: pd.DataFrame
