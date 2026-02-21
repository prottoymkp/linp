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


def _build_purchase_summary(
    fg_result: pd.DataFrame,
    bom: pd.DataFrame,
    rm: pd.DataFrame,
    mode_avail: str,
) -> pd.DataFrame:
    coverage_levels = [25, 50, 75, 100]
    rm_col = "Avail_Stock" if mode_avail == "STOCK" else "Avail_StockPO"

    cap_map = dict(zip(fg_result["FG Code"].astype(str), pd.to_numeric(fg_result["Plan Cap"], errors="coerce").fillna(0.0)))
    avail_map = dict(zip(rm["RM Code"].astype(str), pd.to_numeric(rm[rm_col], errors="coerce").fillna(0.0)))

    rows = []
    for coverage in coverage_levels:
        fg_qty = {fg_code: int(np.floor(cap * coverage / 100.0)) for fg_code, cap in cap_map.items()}
        required = {rm_code: 0.0 for rm_code in avail_map}

        for _, row in bom.iterrows():
            fg_code = str(row["FG Code"])
            rm_code = str(row["RM Code"])
            if rm_code not in required:
                continue
            required[rm_code] += float(row["QtyPerPair"]) * fg_qty.get(fg_code, 0)

        for rm_code, avail in avail_map.items():
            req = required[rm_code]
            rows.append(
                {
                    "Coverage %": coverage,
                    "RM Code": rm_code,
                    "Current Availability": float(avail),
                    "Required Qty": float(req),
                    "Purchase Required": max(float(req - avail), 0.0),
                }
            )

    return pd.DataFrame(rows)


def generate_purchase_planning_scenarios(
    fg: pd.DataFrame,
    cap: pd.DataFrame,
    mode_avail: str,
) -> pd.DataFrame:
    fill_levels = [0.25, 0.50, 0.75, 1.00]

    cap_col = _cap_col(cap)
    margin_col = _margin_col(fg)

    cap_series = pd.to_numeric(cap[cap_col], errors="coerce").fillna(0.0)
    cap_map = dict(zip(cap["FG Code"].astype(str), cap_series))

    margin_series = pd.to_numeric(fg[margin_col], errors="coerce").fillna(0.0)
    margin_map = dict(zip(fg["FG Code"].astype(str), margin_series))

    total_cap_pairs = float(cap_series.sum())
    plan_margin_max = float(sum(cap_val * margin_map.get(fg_code, 0.0) for fg_code, cap_val in cap_map.items()))

    rows = []
    for fill_pct in fill_levels:
        rows.append(
            {
                "target_metric": "PAIRS",
                "fill_pct": fill_pct,
                "target_value": int(np.ceil(fill_pct * total_cap_pairs)),
                "mode_avail": mode_avail,
                "status": "runnable",
            }
        )

        margin_status = "runnable" if plan_margin_max > 0 else "not_run_plan_margin_nonpositive"
        rows.append(
            {
                "target_metric": "MARGIN_AT_PAIR_FILL",
                "fill_pct": fill_pct,
                "target_value": fill_pct * plan_margin_max,
                "mode_avail": mode_avail,
                "status": margin_status,
            }
        )

    return pd.DataFrame(rows)


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
    res["Fill_FG"] = np.where(res["Plan Cap"] == 0, 0.0, res["Opt Qty Total"] / res["Plan Cap"])
    res = res[["FG Code", "Plan Cap", "Opt Qty Phase A", "Opt Qty Phase B", "Opt Qty Total", "Unit Margin", "Total Margin", "Fill_FG"]]

    total_cap_pairs = float(res["Plan Cap"].sum())
    achieved_pairs = float(res["Opt Qty Total"].sum())
    overall_fill_pairs = 0.0 if total_cap_pairs == 0 else achieved_pairs / total_cap_pairs
    plan_margin_max = float((res["Plan Cap"] * res["Unit Margin"]).sum())
    achieved_margin = float((res["Unit Margin"] * res["Opt Qty Total"]).sum())
    achieved_margin_at_pair_fill = overall_fill_pairs * plan_margin_max
    margin_fill_at_pair_fill = (
        0.0 if achieved_margin_at_pair_fill == 0 else achieved_margin / achieved_margin_at_pair_fill
    )

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
                "TotalCapPairs": total_cap_pairs,
                "AchievedPairs": achieved_pairs,
                "OverallFillPairs": overall_fill_pairs,
                "PlanMarginMax": plan_margin_max,
                "AchievedMargin": achieved_margin,
                "AchievedMarginAtPairFill": achieved_margin_at_pair_fill,
                "MarginFillAtPairFill": margin_fill_at_pair_fill,
            }
        ]
    )

    purchase_summary = _build_purchase_summary(res, bom, rm, config.mode_avail) if config.run_purchase_planner else None

    return TwoPhaseResult(res, rm_diag, run_meta, purchase_summary)


def run_optimization(tables: Dict[str, pd.DataFrame], run_purchase_planner: bool = False):
    ctrl = _control_map(tables[CONTROL_DATASET])
    cfg = RunConfig(
        mode_avail=ctrl["Mode_Avail"].upper(),
        objective=ctrl["Objective"].upper(),
        big_m_cap=10**9,
        run_purchase_planner=run_purchase_planner,
    )
    out = run_two_phase(tables, cfg)

    fg_result = out.fg_result.copy()
    rm_diag = out.rm_diagnostic.copy()
    run_meta_df = out.run_meta.copy()

    achieved_pairs = int(pd.to_numeric(fg_result["Opt Qty Total"], errors="coerce").fillna(0).sum())
    achieved_margin = float(pd.to_numeric(fg_result["Total Margin"], errors="coerce").fillna(0.0).sum())
    total_cap = float(pd.to_numeric(fg_result["Plan Cap"], errors="coerce").fillna(0.0).sum())
    implied_pair_fill = float((achieved_pairs / total_cap) * 100.0) if total_cap > 0 else 0.0

    target_metric = "MARGIN_AT_PAIR_FILL" if cfg.objective == "PLAN" else cfg.objective
    phase_b_status = str(run_meta_df.loc[0, "phase_b_status"]) if "phase_b_status" in run_meta_df.columns else "not_run"
    phase_b_method = str(run_meta_df.loc[0, "phase_b_method"]) if "phase_b_method" in run_meta_df.columns else "not_run"

    rm_source = tables[RM_DATASET].copy()
    rm_rate_col = "RM_Rate" if "RM_Rate" in rm_source.columns else None
    rm_source["RM Code"] = rm_source["RM Code"].astype(str)
    rm_source["RM_Rate"] = pd.to_numeric(rm_source[rm_rate_col], errors="coerce").fillna(0.0) if rm_rate_col else 0.0
    rm_source["BuyQty"] = pd.to_numeric(rm_source["BuyQty"], errors="coerce").fillna(0.0) if "BuyQty" in rm_source.columns else 0.0
    rm_source["BuyCost"] = rm_source["BuyQty"] * rm_source["RM_Rate"]

    include_zero_buy_qty = str(ctrl.get("IncludeZeroBuyQty", "FALSE")).upper() in {"TRUE", "1", "YES"}
    purchase_detail = rm_source[["RM Code", "BuyQty", "RM_Rate", "BuyCost"]].copy()
    if not include_zero_buy_qty:
        purchase_detail = purchase_detail[purchase_detail["BuyQty"] > 0]

    purchase_detail.insert(0, "TargetFillPct", implied_pair_fill)
    purchase_detail.insert(0, "TargetMetric", target_metric)
    purchase_detail = purchase_detail[["TargetMetric", "TargetFillPct", "RM Code", "BuyQty", "RM_Rate", "BuyCost"]]

    total_buy_cost = float(pd.to_numeric(purchase_detail["BuyCost"], errors="coerce").fillna(0.0).sum()) if not purchase_detail.empty else 0.0
    purchase_summary = pd.DataFrame(
        [
            {
                "TargetMetric": target_metric,
                "TargetFillPct": implied_pair_fill,
                "TargetValue": achieved_margin if target_metric == "MARGIN_AT_PAIR_FILL" else achieved_pairs,
                "AchievedPairs": achieved_pairs,
                "AchievedMargin": achieved_margin,
                "TotalBuyCost": total_buy_cost,
                "Mode_Avail": cfg.mode_avail,
                "Status": phase_b_status,
                "Method": phase_b_method,
                "ImpliedPairFill": implied_pair_fill,
                "TargetMarginAtImpliedPairFill": achieved_margin,
                "MarginFillAtImpliedPairFill": achieved_margin,
            }
        ]
    )

    meta = out.run_meta.melt(var_name="Key", value_name="Value")
    meta["Value"] = meta["Value"].astype(str)
    return fg_result, rm_diag, meta, purchase_summary, purchase_detail
    if run_purchase_planner:
        return out.fg_result, out.rm_diagnostic, meta, out.purchase_summary
    return out.fg_result, out.rm_diagnostic, meta
