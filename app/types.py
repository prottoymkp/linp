from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class RunConfig:
    mode_avail: str
    objective: str
    big_m_cap: int = 10**9


@dataclass
class SolveOutcome:
    quantities: Dict[str, int]
    objective_value: float
    status: str
    solver: str
    used_fallback: bool
    method: str
    runtime_sec: float


@dataclass
class TwoPhaseResult:
    fg_result: pd.DataFrame
    rm_diagnostic: pd.DataFrame
    run_meta: pd.DataFrame
