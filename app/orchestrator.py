from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from .config import BOM_DATASET, CAP_DATASET, CONTROL_DATASET, FG_DATASET, RM_DATASET
from .model import solve_optimization
from .types import RunConfig, TwoPhaseResult


def _control_map(df: pd.DataFrame):
    if {"Key", "Value"}.issubset(df.columns):
        return dict(zip(df["Key"].astype(str), df["Value"].astype(str)))
    if {"Control Key", "Control Value"}.issubset(df.columns):
        return dict(zip(df["Control Key"].astype(str), df["Control Value"].astype(str)))
    return dict(zip(df["Setting"].astype(str), df["Value"].astype(str)))


def _cap_col(df):
    return "Max Plan Qty" if "Max Plan Qty" in df.columns else "Plan Cap"


def _margin_col(df):
    return "Margin" if "Margin" in df.columns else "Unit Margin"


def run_two_phase(tables: Dict[str, pd.DataFrame], config: RunConfig) -> TwoPhaseResult:
    fg = tables[FG_DATASET].copy()
    bom = tables[BOM_DATASET].copy()
    cap = tables[CAP_DATASET].copy()
    rm = tables[RM_DATASET].copy()

    cap_col = _cap_col(cap)
    margin_col = _margin_col(fg)
    cap_map = dict(zip(cap["FG Code"].astype(str), pd.to_numeric(cap[cap_col], errors="coerce").fillna(0).astype(int)))

    phase_a = solve_optimization(fg, bom, cap, rm, config.mode_avail, config.objective, config.big_m_cap, enforce_caps=True)
    all_caps_hit = all(int(phase_a.quantities.get(code, 0)) >= int(cap_map.get(code, 0)) for code in fg["FG Code"].astype(str))

    phase_b_qty = {c: 0 for c in fg["FG Code"].astype(str)}
    phase_b_executed = False
    phase_b_status = "skipped"
    phase_b_method = "not_run"

    if all_caps_hit:
        rm_col = "Avail_Stock" if config.mode_avail == "STOCK" else "Avail_StockPO"
        rm_res = rm.copy()
        rm_res[rm_col] = pd.to_numeric(rm_res[rm_col], errors="coerce").fillna(0.0)
        for _, row in bom.iterrows():
            fg_code = str(row["FG Code"])
            rm_code = str(row["RM Code"])
            rm_res.loc[rm_res["RM Code"].astype(str) == rm_code, rm_col] -= float(row["QtyPerPair"]) * phase_a.quantities.get(fg_code, 0)
        rm_res[rm_col] = rm_res[rm_col].clip(lower=0)

        phase_b = solve_optimization(fg, bom, cap, rm_res, config.mode_avail, config.objective, config.big_m_cap, enforce_caps=False)
        phase_b_qty = phase_b.quantities
        phase_b_executed = True
        phase_b_status = phase_b.status
        phase_b_method = phase_b.method

    res = fg[["FG Code", margin_col]].copy().rename(columns={margin_col: "Unit Margin"})
    res["Plan Cap"] = res["FG Code"].astype(str).map(cap_map).fillna(0).astype(int)
    res["Opt Qty Phase A"] = res["FG Code"].astype(str).map(phase_a.quantities).fillna(0).astype(int)
    res["Opt Qty Phase B"] = res["FG Code"].astype(str).map(phase_b_qty).fillna(0).astype(int)
    res["Opt Qty Total"] = res["Opt Qty Phase A"] + res["Opt Qty Phase B"]
    res["Total Margin"] = res["Unit Margin"] * res["Opt Qty Total"]
    res = res[["FG Code", "Plan Cap", "Opt Qty Phase A", "Opt Qty Phase B", "Opt Qty Total", "Unit Margin", "Total Margin"]]

    rm_col = "Avail_Stock" if config.mode_avail == "STOCK" else "Avail_StockPO"
    avail_map = dict(zip(rm["RM Code"].astype(str), pd.to_numeric(rm[rm_col], errors="coerce").fillna(0.0)))
    used = {k: 0.0 for k in avail_map}
    qty_total_map = dict(zip(res["FG Code"].astype(str), res["Opt Qty Total"]))
    for _, row in bom.iterrows():
        used[str(row["RM Code"])] += float(row["QtyPerPair"]) * qty_total_map.get(str(row["FG Code"]), 0)

    rm_diag = pd.DataFrame(
        {
            "RM Code": list(avail_map.keys()),
            "mode_avail": config.mode_avail,
            "objective": config.objective,
            "availability_used": [used[k] for k in avail_map],
            "availability_remaining": [avail_map[k] - used[k] for k in avail_map],
        }
    )

    run_meta = pd.DataFrame(
        [
            {
                "run_ts_utc": datetime.now(timezone.utc).isoformat(),
                "phase_a_status": phase_a.status,
                "phase_a_method": phase_a.method,
                "phase_b_status": phase_b_status,
                "phase_b_method": phase_b_method,
                "phase_b_executed": phase_b_executed,
                "all_caps_hit": all_caps_hit,
            }
        ]
    )

    return TwoPhaseResult(res, rm_diag, run_meta)


def run_optimization(tables: Dict[str, pd.DataFrame]):
    ctrl = _control_map(tables[CONTROL_DATASET])
    cfg = RunConfig(mode_avail=ctrl["Mode_Avail"], objective=ctrl["Objective"], big_m_cap=10**9)
    out = run_two_phase(tables, cfg)
    return out.fg_result, out.rm_diagnostic, out.run_meta.rename(columns={"all_caps_hit": "Value"})
