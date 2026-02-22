from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class RunConfig:
    mode_avail: str
    objective: str
    big_m_cap: int = 10**9
    run_purchase_planner: bool = False


@dataclass
class SolveOutcome:
    quantities: Dict[str, int]
    objective_value: float
    status: str
    solver: str
    used_fallback: bool
    method: str
    runtime_sec: float
    heuristic_cutoff_hit: bool = False
    heuristic_iterations: int = 0
    fallback_elapsed_sec: float = 0.0


@dataclass
class TwoPhaseResult:
    fg_result: pd.DataFrame
    rm_diagnostic: pd.DataFrame
    run_meta: pd.DataFrame
    purchase_summary: pd.DataFrame | None = None
