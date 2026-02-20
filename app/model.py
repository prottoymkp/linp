"""Optimization model builder/solver with HiGHS-first and heuristic fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.types import SolveOutcome

try:
    from highspy import Highs
except Exception:  # pragma: no cover
    Highs = None


@dataclass
class ModelData:
    fg_codes: list[str]
    margins: np.ndarray
    caps: np.ndarray
    rm_codes: list[str]
    avail: np.ndarray
    usage: np.ndarray


def build_model_data(fg_df: pd.DataFrame, bom_df: pd.DataFrame, cap_df: pd.DataFrame, rm_df: pd.DataFrame, mode_avail: str, upper_bounds: Optional[Dict[str, int]] = None) -> ModelData:
    fg_codes = list(fg_df["FG Code"])
    rm_codes = list(rm_df["RM Code"])
    fg_idx = {fg: i for i, fg in enumerate(fg_codes)}
    rm_idx = {rm: j for j, rm in enumerate(rm_codes)}

    margins = pd.to_numeric(fg_df["Unit Margin"], errors="coerce").to_numpy(dtype=float)
    cap_map = {r["FG Code"]: int(float(r["Plan Cap"])) for _, r in cap_df.iterrows()}

    if upper_bounds is None:
        caps = np.array([cap_map[fg] for fg in fg_codes], dtype=float)
    else:
        caps = np.array([int(upper_bounds.get(fg, cap_map.get(fg, 0))) for fg in fg_codes], dtype=float)

    avail_col = "Avail_Stock" if mode_avail == "STOCK" else "Avail_StockPO"
    avail = pd.to_numeric(rm_df[avail_col], errors="coerce").to_numpy(dtype=float)

    usage = np.zeros((len(rm_codes), len(fg_codes)), dtype=float)
    for _, row in bom_df.iterrows():
        i = fg_idx[row["FG Code"]]
        j = rm_idx[row["RM Code"]]
        usage[j, i] = float(row["QtyPerPair"])

    return ModelData(fg_codes, margins, caps, rm_codes, avail, usage)


def _heuristic_solve(model: ModelData, objective: str) -> SolveOutcome:
    start = time.perf_counter()
    x = np.zeros(len(model.fg_codes), dtype=int)
    residual = model.avail.copy()

    if objective == "MARGIN":
        scores = model.margins / np.maximum(model.usage.sum(axis=0), 1e-9)
    else:
        scores = 1.0 / np.maximum(model.usage.sum(axis=0), 1e-9)

    order = list(np.argsort(-scores))
    for i in order:
        ub = int(model.caps[i])
        if ub <= 0:
            continue
        col = model.usage[:, i]
        if np.allclose(col, 0):
            add = ub
        else:
            max_add = np.floor(np.min(np.where(col > 0, residual / np.maximum(col, 1e-12), np.inf)))
            add = int(max(0, min(ub, max_add)))
        if add > 0:
            x[i] += add
            residual -= col * add

    objective_value = float(np.dot(model.margins if objective == "MARGIN" else np.ones_like(model.margins), x))
    return SolveOutcome(
        quantities={fg: int(q) for fg, q in zip(model.fg_codes, x)},
        objective_value=objective_value,
        status="heuristic_optimal",
        solver_used="heuristic",
        fallback_used=True,
        method="heuristic_fallback",
        elapsed_time_sec=time.perf_counter() - start,
    )


def _solve_highs_mip(model: ModelData, objective: str, time_limit_sec: int = 20, *, relax_integrality: bool = False) -> SolveOutcome:
    if Highs is None:
        raise RuntimeError("highspy not available")

    start = time.perf_counter()
    highs = Highs()
    highs.setOptionValue("time_limit", float(time_limit_sec))
    highs.setOptionValue("output_flag", False)

    num_var = len(model.fg_codes)
    cost = -(model.margins if objective == "MARGIN" else np.ones(num_var, dtype=float))
    lower = np.zeros(num_var, dtype=float)
    upper = model.caps.astype(float)

    highs.addVars(num_var, lower, upper)
    for i in range(num_var):
        highs.changeColCost(i, float(cost[i]))
        if not relax_integrality:
            highs.changeColIntegrality(i, 1)

    for r in range(len(model.rm_codes)):
        idx = np.where(model.usage[r, :] > 0)[0].astype(np.int32)
        val = model.usage[r, idx].astype(float)
        highs.addRow(-np.inf, float(model.avail[r]), len(idx), idx, val)

    highs.run()
    model_status = str(highs.getModelStatus())
    if "Optimal" not in model_status and "Feasible" not in model_status:
        raise RuntimeError(f"HiGHS MIP failed with status {model_status}")

    sol = highs.getSolution().col_value
    x = np.floor(np.maximum(sol, 0)).astype(int)
    objective_value = float(np.dot(-cost, x))
    return SolveOutcome(
        quantities={fg: int(q) for fg, q in zip(model.fg_codes, x)},
        objective_value=objective_value,
        status=model_status,
        solver_used="highs_lp" if relax_integrality else "highs_mip",
        fallback_used=False,
        method="lp_relaxation" if relax_integrality else "mip",
        elapsed_time_sec=time.perf_counter() - start,
    )


def solve_optimization(fg_df: pd.DataFrame, bom_df: pd.DataFrame, cap_df: pd.DataFrame, rm_df: pd.DataFrame, mode_avail: str, objective: str, upper_bounds: Optional[Dict[str, int]] = None) -> SolveOutcome:
    model = build_model_data(fg_df, bom_df, cap_df, rm_df, mode_avail, upper_bounds)
    try:
        return _solve_highs_mip(model, objective)
    except Exception:
        try:
            lp_outcome = _solve_highs_mip(model, objective, relax_integrality=True)
            rounded = {fg: int(np.floor(v)) for fg, v in lp_outcome.quantities.items()}
            # verify rounded feasibility; if infeasible, fallback to greedy
            vec = np.array([rounded[fg] for fg in model.fg_codes], dtype=float)
            if np.all(model.usage @ vec <= model.avail + 1e-9):
                lp_outcome.quantities = rounded
                lp_outcome.fallback_used = True
                lp_outcome.method = "heuristic_fallback"
                return lp_outcome
        except Exception:
            pass
        return _heuristic_solve(model, objective)
