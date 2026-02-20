"""Two-phase optimization workflow orchestration and result assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

from app.config import BOM_DATASET, CAP_DATASET, FG_DATASET, RM_DATASET
from app.model import solve_optimization
from app.types import RunConfig, RunOutputs
from app.validate import validate_inputs


def _compute_rm_consumption(fg_codes: list[str], bom_df: pd.DataFrame, quantities: Dict[str, int], rm_codes: list[str]) -> pd.Series:
    usage = pd.Series(0.0, index=rm_codes)
    for _, row in bom_df.iterrows():
        usage[row["RM Code"]] += float(row["QtyPerPair"]) * quantities.get(row["FG Code"], 0)
    return usage


def run_two_phase(data: Dict[str, pd.DataFrame], config: RunConfig) -> RunOutputs:
    fg_df = data[FG_DATASET].copy()
    bom_df = data[BOM_DATASET].copy()
    cap_df = data[CAP_DATASET].copy()
    rm_df = data[RM_DATASET].copy()

    phase_a = solve_optimization(fg_df, bom_df, cap_df, rm_df, config.mode_avail, config.objective)

    cap_map = {r["FG Code"]: int(float(r["Plan Cap"])) for _, r in cap_df.iterrows()}
    all_caps_met = all(int(phase_a.quantities.get(fg, 0)) == cap_map[fg] for fg in fg_df["FG Code"])

    if all_caps_met:
        avail_col = "Avail_Stock" if config.mode_avail == "STOCK" else "Avail_StockPO"
        phase_a_usage = _compute_rm_consumption(list(fg_df["FG Code"]), bom_df, phase_a.quantities, list(rm_df["RM Code"]))
        residual_rm = rm_df[["RM Code", "Avail_Stock", "Avail_StockPO"]].copy()
        residual_rm[avail_col] = np.maximum(0, residual_rm[avail_col].astype(float) - phase_a_usage.values)

        upper_bounds = {fg: config.phase_b_upper_bound for fg in fg_df["FG Code"]}
        phase_b = solve_optimization(fg_df, bom_df, cap_df, residual_rm, config.mode_avail, config.objective, upper_bounds=upper_bounds)
    else:
        phase_b = phase_a
        phase_b.quantities = {fg: 0 for fg in fg_df["FG Code"]}
        phase_b.objective_value = 0.0
        phase_b.status = "skipped_phase_b_caps_not_met"
        phase_b.method = "phase_b_skipped"

    fg_rows = []
    for _, row in fg_df.iterrows():
        fg = row["FG Code"]
        cap = cap_map[fg]
        a_qty = int(phase_a.quantities.get(fg, 0))
        b_qty = int(phase_b.quantities.get(fg, 0))
        total = a_qty + b_qty
        margin = float(row["Unit Margin"])
        fg_rows.append(
            {
                "FG Code": fg,
                "Plan Cap": cap,
                "Opt Qty Phase A": a_qty,
                "Opt Qty Phase B": b_qty,
                "Opt Qty Total": total,
                "Unit Margin": margin,
                "Total Margin": total * margin,
            }
        )
    fg_result = pd.DataFrame(fg_rows)

    avail_col = "Avail_Stock" if config.mode_avail == "STOCK" else "Avail_StockPO"
    total_quantities = {fg: int(fg_result.loc[fg_result["FG Code"] == fg, "Opt Qty Total"].iloc[0]) for fg in fg_df["FG Code"]}
    consumed = _compute_rm_consumption(list(fg_df["FG Code"]), bom_df, total_quantities, list(rm_df["RM Code"]))
    availability = rm_df.set_index("RM Code")[avail_col].astype(float)
    rm_diag = pd.DataFrame(
        {
            "RM Code": availability.index,
            "availability_basis": config.mode_avail,
            "availability_used": consumed.values,
            "remaining_availability": (availability - consumed).values,
            "objective": config.objective,
        }
    )

    run_meta = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode_avail": config.mode_avail,
                "objective": config.objective,
                "solver_used": phase_b.solver_used if all_caps_met else phase_a.solver_used,
                "status": phase_b.status if all_caps_met else phase_a.status,
                "elapsed_time_sec": phase_a.elapsed_time_sec + (phase_b.elapsed_time_sec if all_caps_met else 0),
                "fallback_flag": phase_a.fallback_used or (phase_b.fallback_used if all_caps_met else False),
                "method": phase_b.method if all_caps_met else phase_a.method,
                "phase_b_executed": bool(all_caps_met),
            }
        ]
    )

    return RunOutputs(fg_result=fg_result, rm_diagnostic=rm_diag, run_meta=run_meta)


def run_optimization(data: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible facade used by earlier UI versions."""
    config = validate_inputs(data)
    outputs = run_two_phase(data, config)
    return outputs.fg_result, outputs.rm_diagnostic, outputs.run_meta
