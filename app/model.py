from __future__ import annotations

import time
from typing import Dict

import pandas as pd
import pulp

from .types import SolveOutcome


def _fg_margin_col(df: pd.DataFrame) -> str:
    return "Margin" if "Margin" in df.columns else "Unit Margin"


def _cap_col(df: pd.DataFrame) -> str:
    return "Max Plan Qty" if "Max Plan Qty" in df.columns else "Plan Cap"


def solve_optimization(
    fg_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    rm_df: pd.DataFrame,
    mode_avail: str,
    objective: str,
    big_m_cap: int = 10**9,
    enforce_caps: bool = True,
) -> SolveOutcome:
    start = time.time()
    fg = fg_df.copy()
    fg["FG Code"] = fg["FG Code"].astype(str)
    bom = bom_df.copy()
    bom["FG Code"] = bom["FG Code"].astype(str)
    bom["RM Code"] = bom["RM Code"].astype(str)

    margin_col = _fg_margin_col(fg)
    cap_col = _cap_col(cap_df)

    cap_map = dict(zip(cap_df["FG Code"].astype(str), pd.to_numeric(cap_df[cap_col], errors="coerce").fillna(0.0)))
    rm_col = "Avail_Stock" if mode_avail == "STOCK" else "Avail_StockPO"
    rm_avail = dict(zip(rm_df["RM Code"].astype(str), pd.to_numeric(rm_df[rm_col], errors="coerce").fillna(0.0)))

    prob = pulp.LpProblem("fg_rm", pulp.LpMaximize)
    x = {c: pulp.LpVariable(f"x_{c}", lowBound=0, cat=pulp.LpInteger) for c in fg["FG Code"]}

    for code in fg["FG Code"]:
        upper = cap_map.get(code, 0.0) if enforce_caps else big_m_cap
        prob += x[code] <= upper

    for rm_code, avail in rm_avail.items():
        rows = bom[bom["RM Code"] == rm_code]
        prob += pulp.lpSum(float(r["QtyPerPair"]) * x[r["FG Code"]] for _, r in rows.iterrows()) <= float(avail)

    if objective == "MARGIN":
        margins = dict(zip(fg["FG Code"], pd.to_numeric(fg[margin_col], errors="coerce").fillna(0.0)))
        prob += pulp.lpSum(margins[c] * x[c] for c in x)
    else:
        prob += pulp.lpSum(x[c] for c in x)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    quantities = {k: int(round(v.value() or 0.0)) for k, v in x.items()}
    objective_value = float(pulp.value(prob.objective) or 0.0)
    return SolveOutcome(quantities, objective_value, status, "cbc", False, "mip", time.time() - start)
