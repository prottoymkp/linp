from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import numpy as np
import pandas as pd

from .config import BOM_DATASET, CAP_DATASET, CONTROL_DATASET, FG_DATASET, RM_DATASET
from .model import audit_phaseA_solution, solve_optimization, solve_phaseA_lexicographic
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


def _audit_phase_a_final(
    x_final: np.ndarray,
    caps: np.ndarray,
    coeff: np.ndarray,
    rm_upper: np.ndarray,
    x_stage1: np.ndarray | None = None,
    x_stage2: np.ndarray | None = None,
    plan_stage2_feasible: bool = False,
    p_star: int | None = None,
) -> None:
    assert np.allclose(x_final, np.rint(x_final))
    assert np.all(x_final >= -1e-9)
    assert np.all(x_final <= caps + 1e-9)
    assert np.all(coeff @ x_final <= rm_upper + 1e-9)

    if plan_stage2_feasible and x_stage1 is not None and x_stage2 is not None and p_star is not None:
        assert int(np.rint(x_stage2.sum())) == int(p_star)
        assert int(np.rint(x_stage1.sum())) == int(p_star)


def run_two_phase(tables: Dict[str, pd.DataFrame], config: RunConfig) -> TwoPhaseResult:
    fg = tables[FG_DATASET].copy()
    bom = tables[BOM_DATASET].copy()
    cap = tables[CAP_DATASET].copy()
    rm = tables[RM_DATASET].copy()

    cap_col = _cap_col(cap)
    margin_col = _margin_col(fg)
    cap_map = dict(zip(cap["FG Code"].astype(str), pd.to_numeric(cap[cap_col], errors="coerce").fillna(0).astype(int)))

    objective = str(config.objective).upper()

    phase_a_meta: Dict[str, object] = {}
    if objective == "PLAN":
        phase_a, phase_a_meta, audit_payload = solve_phaseA_lexicographic(fg, bom, cap, rm, config.mode_avail, config.big_m_cap)
        stage2_success = str(phase_a_meta.get("stage2_status")) in {"Optimal", "Feasible", "fallback_reallocated"}
        audit_phaseA_solution(audit_payload, stage2_success=stage2_success)
        _audit_phase_a_final(
            x_final=audit_payload["x_stage2"].astype(float),
            caps=audit_payload["caps"].astype(float),
            coeff=audit_payload["coeff"].astype(float),
            rm_upper=audit_payload["rm_upper"].astype(float),
            x_stage1=audit_payload["x_stage1"].astype(float),
            x_stage2=audit_payload["x_stage2"].astype(float),
            plan_stage2_feasible=stage2_success,
            p_star=int(phase_a_meta.get("P_star", 0)),
        )
    elif objective in {"MARGIN", "PAIRS"}:
        phase_a = solve_optimization(fg, bom, cap, rm, config.mode_avail, objective, config.big_m_cap, enforce_caps=True)
        phase_a_meta = {
            "method": phase_a.method,
            "stage1_status": phase_a.status,
            "stage1_solver_used": phase_a.method,
            "stage1_runtime": phase_a.runtime_sec,
            "P_star": "n/a",
            "stage2_status": "n/a",
            "stage2_solver_used": "n/a",
            "stage2_runtime": "n/a",
            "phaseA_final_status": phase_a.status,
        }

        fg_codes = fg["FG Code"].astype(str).tolist()
        qty = np.array([phase_a.quantities.get(code, 0) for code in fg_codes], dtype=float)
        caps = np.array([cap_map.get(code, 0) for code in fg_codes], dtype=float)
        rm_col = "Avail_Stock" if config.mode_avail == "STOCK" else "Avail_StockPO"
        rm_upper = pd.to_numeric(rm[rm_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        rm_codes = rm["RM Code"].astype(str).tolist()
        rm_index = {r: i for i, r in enumerate(rm_codes)}
        fg_index = {f: i for i, f in enumerate(fg_codes)}
        coeff = np.zeros((len(rm_codes), len(fg_codes)), dtype=float)
        for _, row in bom.iterrows():
            f = str(row["FG Code"])
            r = str(row["RM Code"])
            if f in fg_index and r in rm_index:
                coeff[rm_index[r], fg_index[f]] += float(row["QtyPerPair"])
        _audit_phase_a_final(x_final=qty, caps=caps, coeff=coeff, rm_upper=rm_upper)
    else:
        raise ValueError(f"Unsupported objective: {objective}")

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

        phase_b = solve_optimization(fg, bom, cap, rm_res, config.mode_avail, objective if objective in {"MARGIN", "PAIRS"} else "MARGIN", config.big_m_cap, enforce_caps=False)
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
            "objective": objective,
            "availability_used": [used[k] for k in avail_map],
            "availability_remaining": [avail_map[k] - used[k] for k in avail_map],
        }
    )

    run_meta = pd.DataFrame(
        [
            {
                "run_ts_utc": datetime.now(timezone.utc).isoformat(),
                "objective": objective,
                "phase_a_status": phase_a.status,
                "phase_a_method": phase_a_meta.get("method", phase_a.method),
                "stage1_status": phase_a_meta.get("stage1_status", phase_a.status),
                "stage1_solver_used": phase_a_meta.get("stage1_solver_used", phase_a.method),
                "P_star": phase_a_meta.get("P_star", "n/a"),
                "stage1_runtime": phase_a_meta.get("stage1_runtime", phase_a.runtime_sec),
                "stage2_status": phase_a_meta.get("stage2_status", "n/a"),
                "stage2_solver_used": phase_a_meta.get("stage2_solver_used", "n/a"),
                "stage2_runtime": phase_a_meta.get("stage2_runtime", "n/a"),
                "phaseA_final_status": phase_a_meta.get("phaseA_final_status", phase_a.status),
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
    cfg = RunConfig(mode_avail=ctrl["Mode_Avail"].upper(), objective=ctrl["Objective"].upper(), big_m_cap=10**9)
    out = run_two_phase(tables, cfg)
    meta = out.run_meta.melt(var_name="Key", value_name="Value")
    meta["Value"] = meta["Value"].astype(str)
    return out.fg_result, out.rm_diagnostic, meta
