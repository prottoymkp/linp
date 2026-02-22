from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Dict, Iterator

import numpy as np
import pandas as pd

from .config import BOM_DATASET, CAP_DATASET, CONTROL_DATASET, FG_DATASET, RM_DATASET
from .model import audit_phaseA_solution, solve_optimization, solve_phaseA_lexicographic, solve_purchase_plan_pairs_target
from .types import RunConfig, TwoPhaseResult


ProgressCallback = Callable[[str, float, float, str, bool], None]
_HEARTBEAT_INTERVAL_SEC = 3.0


def _notify_progress(
    callback: ProgressCallback | None,
    stage: str,
    stage_progress_pct: float,
    overall_progress_pct: float,
    status_text: str | None = None,
    is_heartbeat: bool = False,
) -> None:
    if callback is None:
        return
    callback(
        stage,
        float(np.clip(stage_progress_pct, 0.0, 100.0)),
        float(np.clip(overall_progress_pct, 0.0, 100.0)),
        status_text or stage,
        is_heartbeat,
    )


def _solver_limits_text() -> str:
    return "time_limit=None, mip_rel_gap=None"


@contextmanager
def _with_solver_heartbeat(
    callback: ProgressCallback | None,
    stage: str,
    stage_progress_pct: float,
    overall_progress_pct: float,
) -> Iterator[None]:
    start = time.monotonic()
    done = threading.Event()

    def _status() -> str:
        elapsed = time.monotonic() - start
        return f"{stage} | elapsed {elapsed:.1f}s | solver limits: {_solver_limits_text()}"

    def _heartbeater() -> None:
        while not done.wait(_HEARTBEAT_INTERVAL_SEC):
            _notify_progress(
                callback,
                stage,
                stage_progress_pct,
                overall_progress_pct,
                status_text=_status(),
                is_heartbeat=True,
            )

    hb_thread = threading.Thread(target=_heartbeater, daemon=True)
    hb_thread.start()
    _notify_progress(
        callback,
        stage,
        stage_progress_pct,
        overall_progress_pct,
        status_text=_status(),
        is_heartbeat=False,
    )
    try:
        yield
    finally:
        done.set()
        hb_thread.join(timeout=0.2)


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



def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"auto", "none", "null", "unset", "default"}:
        return None
    return int(float(text))


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "unset", "default"}:
        return None
    return float(text)

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


def run_two_phase(
    tables: Dict[str, pd.DataFrame],
    config: RunConfig,
    progress_callback: ProgressCallback | None = None,
) -> TwoPhaseResult:
    _notify_progress(progress_callback, "Phase A - preparing optimization", 5, 10)
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
        with _with_solver_heartbeat(progress_callback, "Phase A - lexicographic solve", 45, 25):
            phase_a, phase_a_meta, audit_payload = solve_phaseA_lexicographic(fg, bom, cap, rm, config.mode_avail, config.big_m_cap)
        _notify_progress(progress_callback, "Phase A - lexicographic solve", 45, 25)
        solver_options = {}
        if config.threads is not None:
            solver_options["threads"] = config.threads
        if config.mip_rel_gap is not None:
            solver_options["mip_rel_gap"] = config.mip_rel_gap
        if config.time_limit_sec is not None:
            solver_options["time_limit_sec"] = config.time_limit_sec
        phase_a, phase_a_meta, audit_payload = solve_phaseA_lexicographic(
            fg, bom, cap, rm, config.mode_avail, config.big_m_cap, **solver_options
        )
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
        with _with_solver_heartbeat(progress_callback, f"Phase A - solving for {objective}", 45, 25):
            phase_a = solve_optimization(fg, bom, cap, rm, config.mode_avail, objective, config.big_m_cap, enforce_caps=True)
        _notify_progress(progress_callback, f"Phase A - solving for {objective}", 45, 25)
        solver_options = {}
        if config.threads is not None:
            solver_options["threads"] = config.threads
        if config.mip_rel_gap is not None:
            solver_options["mip_rel_gap"] = config.mip_rel_gap
        if config.time_limit_sec is not None:
            solver_options["time_limit_sec"] = config.time_limit_sec
        phase_a = solve_optimization(
            fg, bom, cap, rm, config.mode_avail, objective, config.big_m_cap, enforce_caps=True, **solver_options
        )
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
            "heuristic_cutoff_hit": phase_a.heuristic_cutoff_hit,
            "heuristic_iterations": phase_a.heuristic_iterations,
            "fallback_passes": "n/a",
            "fallback_swaps": "n/a",
            "fallback_elapsed_sec": phase_a.fallback_elapsed_sec,
            "cutoff_reason": "none",
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

    _notify_progress(progress_callback, "Phase A - completed", 100, 45)

    all_caps_hit = all(int(phase_a.quantities.get(code, 0)) >= int(cap_map.get(code, 0)) for code in fg["FG Code"].astype(str))

    phase_b_qty = {c: 0 for c in fg["FG Code"].astype(str)}
    phase_b_executed = False
    phase_b_status = "skipped"
    phase_b_method = "not_run"

    if all_caps_hit:
        _notify_progress(progress_callback, "Phase B - re-optimizing remaining inventory", 50, 50)
        rm_col = "Avail_Stock" if config.mode_avail == "STOCK" else "Avail_StockPO"
        rm_res = rm.copy()
        rm_res[rm_col] = pd.to_numeric(rm_res[rm_col], errors="coerce").fillna(0.0)
        for _, row in bom.iterrows():
            fg_code = str(row["FG Code"])
            rm_code = str(row["RM Code"])
            rm_res.loc[rm_res["RM Code"].astype(str) == rm_code, rm_col] -= float(row["QtyPerPair"]) * phase_a.quantities.get(fg_code, 0)
        rm_res[rm_col] = rm_res[rm_col].clip(lower=0)

        with _with_solver_heartbeat(progress_callback, "Phase B - re-optimizing remaining inventory", 50, 50):
            phase_b = solve_optimization(fg, bom, cap, rm_res, config.mode_avail, objective if objective in {"MARGIN", "PAIRS"} else "MARGIN", config.big_m_cap, enforce_caps=False)
        solver_options = {}
        if config.threads is not None:
            solver_options["threads"] = config.threads
        if config.mip_rel_gap is not None:
            solver_options["mip_rel_gap"] = config.mip_rel_gap
        if config.time_limit_sec is not None:
            solver_options["time_limit_sec"] = config.time_limit_sec
        phase_b = solve_optimization(
            fg, bom, cap, rm_res, config.mode_avail, objective if objective in {"MARGIN", "PAIRS"} else "MARGIN", config.big_m_cap, enforce_caps=False, **solver_options
        )
        phase_b_qty = phase_b.quantities
        phase_b_executed = True
        phase_b_status = phase_b.status
        phase_b_method = phase_b.method

    _notify_progress(progress_callback, "Phase B - completed", 100, 60)

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
                "heuristic_cutoff_hit": phase_a_meta.get("heuristic_cutoff_hit", phase_a.heuristic_cutoff_hit),
                "heuristic_iterations": phase_a_meta.get("heuristic_iterations", phase_a.heuristic_iterations),
                "fallback_passes": phase_a_meta.get("fallback_passes", "n/a"),
                "fallback_swaps": phase_a_meta.get("fallback_swaps", "n/a"),
                "fallback_elapsed_sec": phase_a_meta.get("fallback_elapsed_sec", phase_a.fallback_elapsed_sec),
                "cutoff_reason": phase_a_meta.get("cutoff_reason", "none"),
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


def run_optimization(
    tables: Dict[str, pd.DataFrame],
    run_purchase_planner: bool = False,
    progress_callback: ProgressCallback | None = None,
    threads: int | None = None,
    mip_rel_gap: float | None = None,
    time_limit_sec: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _notify_progress(progress_callback, "Initializing optimization", 0, 0)
    ctrl = _control_map(tables[CONTROL_DATASET])
    control_threads = _parse_optional_int(ctrl.get("threads"))
    control_mip_rel_gap = _parse_optional_float(ctrl.get("mip_rel_gap"))
    control_time_limit_sec = _parse_optional_float(ctrl.get("time_limit_sec"))

    cfg = RunConfig(
        mode_avail=ctrl["Mode_Avail"].upper(),
        objective=ctrl["Objective"].upper(),
        big_m_cap=10**9,
        run_purchase_planner=run_purchase_planner,
        threads=threads if threads is not None else control_threads,
        mip_rel_gap=mip_rel_gap if mip_rel_gap is not None else control_mip_rel_gap,
        time_limit_sec=time_limit_sec if time_limit_sec is not None else control_time_limit_sec,
    )
    out = run_two_phase(tables, cfg, progress_callback=progress_callback)
    _notify_progress(progress_callback, "Core optimization finished", 100, 65)

    fg_result = out.fg_result.copy()
    rm_diag = out.rm_diagnostic.copy()
    run_meta_df = out.run_meta.copy()

    achieved_pairs = int(pd.to_numeric(fg_result["Opt Qty Total"], errors="coerce").fillna(0).sum())
    achieved_margin = float(pd.to_numeric(fg_result["Total Margin"], errors="coerce").fillna(0.0).sum())

    rm_source = tables[RM_DATASET].copy()
    rm_source["RM Code"] = rm_source["RM Code"].astype(str)
    rm_col = "Avail_Stock" if cfg.mode_avail == "STOCK" else "Avail_StockPO"
    rm_source["AvailBase"] = pd.to_numeric(rm_source[rm_col], errors="coerce").fillna(0.0)

    targets = [0.25, 0.50, 0.75, 1.00]
    purchase_rows = []
    detail_rows = []
    purchase_sheets: Dict[int, Dict[str, pd.DataFrame]] = {}

    purchase_plan_status = "not_run"
    purchase_solver = "n/a"

    missing_rm_rate = run_purchase_planner and "RM_Rate" not in rm_source.columns
    if "RM_Rate" not in rm_source.columns:
        rm_source["RM_Rate"] = 0.0
    else:
        rm_source["RM_Rate"] = pd.to_numeric(rm_source["RM_Rate"], errors="coerce").fillna(0.0)
    if missing_rm_rate:
        purchase_plan_status = "skipped_missing_rm_rate"

    cap_sum = float(pd.to_numeric(fg_result["Plan Cap"], errors="coerce").fillna(0).sum())

    for idx, target in enumerate(targets):
        pct = int(target * 100)
        target_pairs = int(np.ceil(target * cap_sum))
        stage_base = idx / len(targets)
        _notify_progress(progress_callback, f"Purchase planning {pct}% target", stage_base * 100, 70 + (20 * stage_base))

        if (not run_purchase_planner) or missing_rm_rate:
            status = "skipped_missing_rm_rate" if missing_rm_rate else "not_run"
            method = "not_run"
            x_map = {str(code): 0 for code in fg_result["FG Code"].astype(str)}
            buy_map = {str(code): 0.0 for code in rm_source["RM Code"].astype(str)}
            total_buy_cost = 0.0
            achieved = 0
            margin = 0.0
        else:
            with _with_solver_heartbeat(
                progress_callback,
                f"Purchase planning {pct}% target - solver",
                min(stage_base * 100 + 50.0, 99.0),
                70 + (20 * stage_base),
            ):
                solve = solve_purchase_plan_pairs_target(
                    fg_df=tables[FG_DATASET],
                    bom_df=tables[BOM_DATASET],
                    cap_df=tables[CAP_DATASET],
                    rm_df=rm_source[["RM Code", "Avail_Stock", "Avail_StockPO", "RM_Rate"]],
                    mode_avail=cfg.mode_avail,
                    target_fill_pct=target,
                )
            solver_options = {}
            if cfg.threads is not None:
                solver_options["threads"] = cfg.threads
            if cfg.mip_rel_gap is not None:
                solver_options["mip_rel_gap"] = cfg.mip_rel_gap
            if cfg.time_limit_sec is not None:
                solver_options["time_limit_sec"] = cfg.time_limit_sec
            solve = solve_purchase_plan_pairs_target(
                fg_df=tables[FG_DATASET],
                bom_df=tables[BOM_DATASET],
                cap_df=tables[CAP_DATASET],
                rm_df=rm_source[["RM Code", "Avail_Stock", "Avail_StockPO", "RM_Rate"]],
                mode_avail=cfg.mode_avail,
                target_fill_pct=target,
                **solver_options,
            )
            status = str(solve["status"])
            method = str(solve["method"])
            x_map = {str(k): int(v) for k, v in solve["x"].items()}
            buy_map = {str(k): float(v) for k, v in solve["buy"].items()}
            total_buy_cost = float(solve["total_buy_cost"])
            achieved = int(sum(x_map.values()))
            margin_map = dict(zip(fg_result["FG Code"].astype(str), pd.to_numeric(fg_result["Unit Margin"], errors="coerce").fillna(0.0)))
            margin = float(sum(margin_map.get(code, 0.0) * qty for code, qty in x_map.items()))
            purchase_plan_status = "ran"
            purchase_solver = str(solve.get("method", "highs_mip"))

            # audits
            caps_map = dict(zip(fg_result["FG Code"].astype(str), pd.to_numeric(fg_result["Plan Cap"], errors="coerce").fillna(0).astype(int)))
            for code, qty in x_map.items():
                assert float(qty).is_integer()
                assert 0 <= qty <= int(caps_map.get(code, 0))
            assert achieved >= target_pairs or status not in {"Optimal", "Feasible"}
            bom = tables[BOM_DATASET]
            usage = {rm: 0.0 for rm in rm_source["RM Code"].astype(str)}
            for _, row in bom.iterrows():
                fg_code = str(row["FG Code"])
                rm_code = str(row["RM Code"])
                if rm_code in usage:
                    usage[rm_code] += float(row["QtyPerPair"]) * x_map.get(fg_code, 0)
            for rm_code, used in usage.items():
                avail = float(rm_source.loc[rm_source["RM Code"] == rm_code, "AvailBase"].iloc[0])
                buy = float(buy_map.get(rm_code, 0.0))
                assert used <= avail + buy + 1e-6
            calc_buy_cost = float(sum(float(buy_map.get(rm, 0.0)) * float(rm_source.loc[rm_source["RM Code"] == rm, "RM_Rate"].iloc[0]) for rm in usage))
            assert abs(calc_buy_cost - total_buy_cost) <= 1e-5

        fg_sheet = fg_result[["FG Code", "Plan Cap", "Unit Margin"]].copy()
        fg_sheet = fg_sheet.rename(columns={"Plan Cap": "Cap", "Unit Margin": "UnitMargin"})
        fg_sheet["TargetPairs"] = target_pairs
        fg_sheet["OptQty"] = fg_sheet["FG Code"].astype(str).map(x_map).fillna(0).astype(int)
        fg_sheet["Shortfall"] = (fg_sheet["Cap"] - fg_sheet["OptQty"]).clip(lower=0)
        fg_sheet["TotalMargin"] = fg_sheet["OptQty"] * fg_sheet["UnitMargin"]
        fg_sheet = fg_sheet[["FG Code", "Cap", "TargetPairs", "OptQty", "Shortfall", "UnitMargin", "TotalMargin"]]

        rm_sheet = rm_source[["RM Code", "AvailBase", "RM_Rate"]].copy()
        usage_map = {rm: 0.0 for rm in rm_sheet["RM Code"].astype(str)}
        for _, brow in tables[BOM_DATASET].iterrows():
            fg_code = str(brow["FG Code"])
            rm_code = str(brow["RM Code"])
            if rm_code in usage_map:
                usage_map[rm_code] += float(brow["QtyPerPair"]) * x_map.get(fg_code, 0)
        rm_sheet["UsedAtPlan"] = rm_sheet["RM Code"].astype(str).map(usage_map).fillna(0.0)
        rm_sheet["BuyQty"] = rm_sheet["RM Code"].astype(str).map(buy_map).fillna(0.0)
        rm_sheet["BuyCost"] = rm_sheet["BuyQty"] * rm_sheet["RM_Rate"]
        rm_sheet["RemainingAfterBuy"] = rm_sheet["AvailBase"] + rm_sheet["BuyQty"] - rm_sheet["UsedAtPlan"]
        rm_sheet = rm_sheet[["RM Code", "AvailBase", "RM_Rate", "UsedAtPlan", "BuyQty", "BuyCost", "RemainingAfterBuy"]]

        purchase_sheets[pct] = {"fg": fg_sheet, "rm": rm_sheet}

        purchase_rows.append(
            {
                "TargetFillPct": float(pct),
                "TargetPairs": target_pairs,
                "AchievedPairs": int(achieved),
                "AchievedMargin": float(margin),
                "TotalBuyCost": float(total_buy_cost),
                "Mode_Avail": cfg.mode_avail,
                "Status": status,
                "Method": method,
            }
        )
        for _, r in rm_sheet.iterrows():
            detail_rows.append({"TargetFillPct": float(pct), "RM Code": r["RM Code"], "BuyQty": r["BuyQty"], "RM_Rate": r["RM_Rate"], "BuyCost": r["BuyCost"]})

        stage_end = (idx + 1) / len(targets)
        _notify_progress(progress_callback, f"Purchase planning {pct}% target", 100, 70 + (20 * stage_end))

    purchase_summary = pd.DataFrame(purchase_rows)
    purchase_detail = pd.DataFrame(detail_rows, columns=["TargetFillPct", "RM Code", "BuyQty", "RM_Rate", "BuyCost"])
    include_zero_buy_qty = str(ctrl.get("IncludeZeroBuyQty", "FALSE")).upper() in {"TRUE", "1", "YES"}
    if (not include_zero_buy_qty) or missing_rm_rate:
        purchase_detail = purchase_detail[purchase_detail["BuyQty"] > 0] if not purchase_detail.empty else purchase_detail

    run_meta_df.loc[0, "purchase_plan_status"] = purchase_plan_status
    run_meta_df.loc[0, "purchase_targets"] = "25,50,75,100"
    run_meta_df.loc[0, "purchase_mode_avail"] = cfg.mode_avail
    run_meta_df.loc[0, "purchase_solver"] = purchase_solver

    meta = out.run_meta.melt(var_name="Key", value_name="Value")
    meta["Value"] = meta["Value"].fillna("n/a").astype(str)
    extra_meta = pd.DataFrame(
        {
            "Key": ["purchase_plan_status", "purchase_targets", "purchase_mode_avail", "purchase_solver"],
            "Value": [str(purchase_plan_status), "25,50,75,100", cfg.mode_avail, str(purchase_solver)],
        }
    )
    meta = pd.concat([meta, extra_meta], ignore_index=True)

    purchase_detail.attrs["purchase_target_sheets"] = purchase_sheets
    _notify_progress(progress_callback, "Completed", 100, 100)
    return fg_result, rm_diag, meta, purchase_summary, purchase_detail
