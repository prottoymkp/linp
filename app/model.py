from __future__ import annotations

import time
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from highspy import (
    Highs,
    HighsModel,
    HighsLp,
    HighsSparseMatrix,
    HighsVarType,
    MatrixFormat,
    ObjSense,
    kHighsInf,
)

from .types import SolveOutcome


def _fg_margin_col(df: pd.DataFrame) -> str:
    return "Margin" if "Margin" in df.columns else "Unit Margin"


def _cap_col(df: pd.DataFrame) -> str:
    return "Max Plan Qty" if "Max Plan Qty" in df.columns else "Plan Cap"


def _rm_rate_col(df: pd.DataFrame) -> str:
    return "RM_Rate" if "RM_Rate" in df.columns else "RM Rate"


def _status_feasible(status: str) -> bool:
    return status in {"Optimal", "Feasible"}


def _build_model_inputs(
    fg_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    rm_df: pd.DataFrame,
    mode_avail: str,
    enforce_caps: bool,
    big_m_cap: int,
) -> Dict[str, object]:
    fg_codes = fg_df["FG Code"].astype(str).tolist()
    n = len(fg_codes)

    cap_col = _cap_col(cap_df)
    cap_map = dict(zip(cap_df["FG Code"].astype(str), pd.to_numeric(cap_df[cap_col], errors="coerce").fillna(0.0)))
    caps = np.array([cap_map.get(code, 0.0) if enforce_caps else float(big_m_cap) for code in fg_codes], dtype=np.float64)

    rm_col = "Avail_Stock" if mode_avail == "STOCK" else "Avail_StockPO"
    rm_avail_map = dict(zip(rm_df["RM Code"].astype(str), pd.to_numeric(rm_df[rm_col], errors="coerce").fillna(0.0)))
    rm_codes = list(rm_avail_map.keys())
    m = len(rm_codes)
    rm_index = {rm: i for i, rm in enumerate(rm_codes)}
    fg_index = {fg: i for i, fg in enumerate(fg_codes)}

    coeff = np.zeros((m, n), dtype=np.float64)
    for _, row in bom_df.iterrows():
        fg_code = str(row["FG Code"])
        rm_code = str(row["RM Code"])
        if fg_code in fg_index and rm_code in rm_index:
            coeff[rm_index[rm_code], fg_index[fg_code]] += float(row["QtyPerPair"])

    col_entries: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for j in range(n):
        nz_rows = np.nonzero(coeff[:, j])[0]
        col_entries[j] = [(int(r), float(coeff[r, j])) for r in nz_rows]

    row_lower = np.full(m, -kHighsInf, dtype=np.float64)
    row_upper = np.array([float(rm_avail_map[rm]) for rm in rm_codes], dtype=np.float64)

    return {
        "fg_codes": fg_codes,
        "caps": caps,
        "coeff": coeff,
        "col_entries": col_entries,
        "row_lower": row_lower,
        "row_upper": row_upper,
        "num_rows": m,
    }


def _build_csc(col_entries: List[List[Tuple[int, float]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = [0]
    index: List[int] = []
    value: List[float] = []
    for entries in col_entries:
        for r, v in entries:
            index.append(r)
            value.append(v)
        starts.append(len(index))
    return (
        np.array(starts, dtype=np.int32),
        np.array(index, dtype=np.int32),
        np.array(value, dtype=np.float64),
    )


def _solve_highs(
    col_cost: np.ndarray,
    col_lower: np.ndarray,
    col_upper: np.ndarray,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    col_entries: List[List[Tuple[int, float]]],
    integer: bool = False,
    integer_col_count: int | None = None,
    var_types: np.ndarray | None = None,
    sense: ObjSense | str = ObjSense.kMaximize,
    threads: int | None = None,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
) -> Tuple[str, np.ndarray, float, float]:
    start = time.time()
    lp = HighsLp()
    lp.num_col_ = int(len(col_cost))
    lp.num_row_ = int(len(row_lower))
    lp.col_cost_ = col_cost.astype(np.float64)
    lp.col_lower_ = col_lower.astype(np.float64)
    lp.col_upper_ = col_upper.astype(np.float64)
    lp.row_lower_ = row_lower.astype(np.float64)
    lp.row_upper_ = row_upper.astype(np.float64)

    astart, aindex, avalue = _build_csc(col_entries)
    mat = HighsSparseMatrix()
    mat.format_ = MatrixFormat.kColwise
    mat.start_ = astart
    mat.index_ = aindex
    mat.value_ = avalue
    lp.a_matrix_ = mat

    model = HighsModel()
    model.lp_ = lp

    highs = Highs()
    highs.setOptionValue("output_flag", False)
    if threads is not None:
        highs.setOptionValue("threads", int(threads))
    else:
        cpu_threads = os.cpu_count()
        if cpu_threads and cpu_threads > 0:
            highs.setOptionValue("threads", int(cpu_threads))
    if time_limit is not None:
        highs.setOptionValue("time_limit", float(time_limit))
    if mip_rel_gap is not None:
        highs.setOptionValue("mip_rel_gap", float(mip_rel_gap))
    highs.passModel(model)
    highs.changeObjectiveSense(ObjSense.kMaximize)

    if integer:
        integer_count = len(col_cost) if integer_col_count is None else int(integer_col_count)
        idx = np.arange(integer_count, dtype=np.int32)
        types = np.array([HighsVarType.kInteger] * integer_count)
        highs.changeColsIntegrality(integer_count, idx, types)
    if isinstance(sense, ObjSense):
        obj_sense = sense
    else:
        norm_sense = str(sense).strip().lower()
        if norm_sense == "min":
            obj_sense = ObjSense.kMinimize
        elif norm_sense == "max":
            obj_sense = ObjSense.kMaximize
        else:
            raise ValueError(f"Unsupported objective sense: {sense}")
    highs.changeObjectiveSense(obj_sense)

    if var_types is not None:
        if len(var_types) != len(col_cost):
            raise ValueError("var_types length must match number of columns")
        idx = np.arange(len(col_cost), dtype=np.int32)
        highs.changeColsIntegrality(len(col_cost), idx, np.asarray(var_types))

    highs.run()
    status = highs.modelStatusToString(highs.getModelStatus())
    sol = highs.getSolution()
    col_value = np.array(sol.col_value[: len(col_cost)], dtype=np.float64)
    return status, col_value, float(highs.getObjectiveValue()), time.time() - start


def _rm_usage(coeff: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coeff @ x


def _greedy_fill(
    base_x: np.ndarray,
    target_total: int,
    caps: np.ndarray,
    coeff: np.ndarray,
    rm_upper: np.ndarray,
    scores: np.ndarray,
    max_iterations: int = 100_000,
    max_wall_time_sec: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    start = time.time()
    x = base_x.astype(int).copy()
    usage = _rm_usage(coeff, x.astype(np.float64))
    order = np.argsort(-scores)
    iterations = 0
    cutoff_hit = False
    cutoff_reason = "none"

    while int(x.sum()) < target_total:
        if iterations >= max_iterations:
            cutoff_hit = True
            cutoff_reason = "max_iterations"
            break
        if (time.time() - start) >= max_wall_time_sec:
            cutoff_hit = True
            cutoff_reason = "max_wall_time"
            break
        moved = False
        for j in order:
            if x[j] >= int(caps[j]):
                continue
            next_usage = usage + coeff[:, j]
            if np.all(next_usage <= rm_upper + 1e-9):
                x[j] += 1
                usage = next_usage
                moved = True
                break
        iterations += 1
        if not moved:
            break
    return x, {
        "iterations": iterations,
        "heuristic_cutoff_hit": cutoff_hit,
        "cutoff_reason": cutoff_reason,
        "elapsed_sec": time.time() - start,
    }


def _fallback_stage2_reallocate(
    x_stage1: np.ndarray,
    margins: np.ndarray,
    caps: np.ndarray,
    coeff: np.ndarray,
    rm_upper: np.ndarray,
    max_passes: int = 2_000,
    max_swaps: int = 20_000,
    max_wall_time_sec: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    start = time.time()
    x = x_stage1.astype(int).copy()
    usage = _rm_usage(coeff, x.astype(np.float64))
    residual = rm_upper - usage

    low_to_high = np.argsort(margins)
    high_to_low = np.argsort(-margins)

    improved = True
    passes = 0
    swaps = 0
    cutoff_hit = False
    cutoff_reason = "none"
    while improved:
        if passes >= max_passes:
            cutoff_hit = True
            cutoff_reason = "max_passes"
            break
        if swaps >= max_swaps:
            cutoff_hit = True
            cutoff_reason = "max_swaps"
            break
        if (time.time() - start) >= max_wall_time_sec:
            cutoff_hit = True
            cutoff_reason = "max_wall_time"
            break
        improved = False
        passes += 1
        for j in high_to_low:
            if x[j] >= int(caps[j]):
                continue
            for k in low_to_high:
                if margins[j] <= margins[k] or x[k] <= 0:
                    continue
                delta = coeff[:, j] - coeff[:, k]
                if np.all(residual - delta >= -1e-9):
                    x[j] += 1
                    x[k] -= 1
                    residual -= delta
                    improved = True
                    swaps += 1
                    break
            if improved:
                break
    return x, {
        "passes": passes,
        "swaps": swaps,
        "heuristic_cutoff_hit": cutoff_hit,
        "cutoff_reason": cutoff_reason,
        "elapsed_sec": time.time() - start,
    }


def _solve_single_objective(
    model_inputs: Dict[str, object],
    objective: np.ndarray,
    var_types: np.ndarray | None = None,
    sense: ObjSense | str = ObjSense.kMaximize,
    threads: int | None = None,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
) -> Tuple[str, np.ndarray, float, float]:
    n = len(model_inputs["fg_codes"])
    return _solve_highs(
        col_cost=objective,
        col_lower=np.zeros(n, dtype=np.float64),
        col_upper=model_inputs["caps"],
        row_lower=model_inputs["row_lower"],
        row_upper=model_inputs["row_upper"],
        col_entries=model_inputs["col_entries"],
        var_types=var_types,
        sense=sense,
        threads=threads,
        time_limit=time_limit,
        mip_rel_gap=mip_rel_gap,
    )


def solve_phaseA_lexicographic(
    fg_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    rm_df: pd.DataFrame,
    mode_avail: str,
    big_m_cap: int = 10**9,
    threads: int | None = None,
    mip_rel_gap: float | None = None,
    time_limit_sec: float | None = None,
) -> Tuple[SolveOutcome, Dict[str, object], Dict[str, np.ndarray]]:
    model_inputs = _build_model_inputs(fg_df, bom_df, cap_df, rm_df, mode_avail, enforce_caps=True, big_m_cap=big_m_cap)
    fg_codes = model_inputs["fg_codes"]
    n = len(fg_codes)
    margins = pd.to_numeric(fg_df[_fg_margin_col(fg_df)], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    meta: Dict[str, object] = {
        "stage1_status": "not_run",
        "stage1_solver_used": "highs_mip",
        "P_star": 0,
        "stage1_runtime": 0.0,
        "stage2_status": "not_run",
        "stage2_solver_used": "highs_mip",
        "stage2_runtime": 0.0,
        "phaseA_final_status": "not_run",
        "method": "lex_mip",
        "heuristic_cutoff_hit": False,
        "heuristic_iterations": 0,
        "fallback_passes": 0,
        "fallback_swaps": 0,
        "fallback_elapsed_sec": 0.0,
        "cutoff_reason": "none",
    }

    # Stage 1: maximize pairs
    mip_var_types = np.array([HighsVarType.kInteger] * n)
    lp_var_types = np.array([HighsVarType.kContinuous] * n)

    s1_status, s1_vals, _, s1_runtime = _solve_single_objective(
        model_inputs,
        np.ones(n, dtype=np.float64),
        var_types=mip_var_types,
        sense="max",
        threads=threads,
        time_limit=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
    )
    meta["stage1_status"] = s1_status
    meta["stage1_runtime"] = s1_runtime
    x1 = np.maximum(np.rint(s1_vals), 0).astype(int)

    if not _status_feasible(s1_status):
        lp_status, lp_vals, _, _ = _solve_single_objective(
            model_inputs,
            np.ones(n, dtype=np.float64),
            var_types=lp_var_types,
            sense="max",
            threads=threads,
            time_limit=time_limit_sec,
            mip_rel_gap=mip_rel_gap,
        )
        base = np.maximum(np.floor(lp_vals), 0).astype(int) if _status_feasible(lp_status) else np.zeros(n, dtype=int)
        x1, greedy_meta = _greedy_fill(
            base,
            int(base.sum()) + int(model_inputs["caps"].sum()),
            model_inputs["caps"],
            model_inputs["coeff"],
            model_inputs["row_upper"],
            np.ones(n, dtype=np.float64),
        )
        meta["stage1_status"] = f"fallback_{lp_status}"
        meta["stage1_solver_used"] = "lp+greedy"
        meta["method"] = "lex_lp+greedy"
        meta["heuristic_iterations"] = int(greedy_meta["iterations"])
        meta["heuristic_cutoff_hit"] = bool(greedy_meta["heuristic_cutoff_hit"])
        meta["fallback_elapsed_sec"] = float(greedy_meta["elapsed_sec"])
        meta["cutoff_reason"] = str(greedy_meta["cutoff_reason"])

    p_star = int(x1.sum())
    meta["P_star"] = p_star

    if p_star <= 0:
        quantities = {code: 0 for code in fg_codes}
        out = SolveOutcome(
            quantities,
            0.0,
            str(meta["stage1_status"]),
            "highs",
            True,
            str(meta["method"]),
            float(meta["stage1_runtime"]),
            heuristic_cutoff_hit=bool(meta["heuristic_cutoff_hit"]),
            heuristic_iterations=int(meta["heuristic_iterations"]),
            fallback_elapsed_sec=float(meta["fallback_elapsed_sec"]),
        )
        meta["stage2_status"] = "skipped_no_pairs"
        meta["phaseA_final_status"] = str(meta["stage1_status"])
        return out, meta, {"x_stage1": x1, "x_stage2": np.zeros_like(x1)}

    # Stage 2: maximize margin with sum(x)=P*
    col_entries_stage2 = [entries + [(int(model_inputs["num_rows"]), 1.0)] for entries in model_inputs["col_entries"]]
    row_lower2 = np.concatenate([model_inputs["row_lower"], np.array([float(p_star)], dtype=np.float64)])
    row_upper2 = np.concatenate([model_inputs["row_upper"], np.array([float(p_star)], dtype=np.float64)])

    s2_status, s2_vals, s2_obj, s2_runtime = _solve_highs(
        col_cost=margins,
        col_lower=np.zeros(n, dtype=np.float64),
        col_upper=model_inputs["caps"],
        row_lower=row_lower2,
        row_upper=row_upper2,
        col_entries=col_entries_stage2,
        var_types=mip_var_types,
        sense="max",
        threads=threads,
        time_limit=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
    )
    meta["stage2_status"] = s2_status
    meta["stage2_runtime"] = s2_runtime

    if _status_feasible(s2_status):
        x2 = np.maximum(np.rint(s2_vals), 0).astype(int)
        phase_status = s2_status
    else:
        x2 = x1.copy()
        phase_status = str(meta["stage1_status"])
        meta["method"] = "lex_lp+greedy" if meta["stage1_solver_used"] != "highs_mip" else "lex_mip"
        if meta["stage1_solver_used"] != "highs_mip":
            x2_alt, reallocate_meta = _fallback_stage2_reallocate(
                x1,
                margins,
                model_inputs["caps"],
                model_inputs["coeff"],
                model_inputs["row_upper"],
            )
            meta["fallback_passes"] = int(reallocate_meta["passes"])
            meta["fallback_swaps"] = int(reallocate_meta["swaps"])
            meta["fallback_elapsed_sec"] = float(reallocate_meta["elapsed_sec"])
            meta["heuristic_cutoff_hit"] = bool(meta["heuristic_cutoff_hit"] or reallocate_meta["heuristic_cutoff_hit"])
            if bool(reallocate_meta["heuristic_cutoff_hit"]):
                meta["cutoff_reason"] = str(reallocate_meta["cutoff_reason"])
            if int(x2_alt.sum()) == p_star:
                x2 = x2_alt
                meta["stage2_status"] = "fallback_reallocated"
                meta["stage2_solver_used"] = "greedy_reallocate"
            else:
                meta["stage2_status"] = "fallback_no_lex_stage2"

    quantities = {code: int(x2[i]) for i, code in enumerate(fg_codes)}
    out = SolveOutcome(
        quantities,
        float(s2_obj),
        phase_status,
        "highs",
        meta["stage1_solver_used"] != "highs_mip",
        str(meta["method"]),
        s1_runtime + s2_runtime,
        heuristic_cutoff_hit=bool(meta["heuristic_cutoff_hit"]),
        heuristic_iterations=int(meta["heuristic_iterations"]),
        fallback_elapsed_sec=float(meta["fallback_elapsed_sec"]),
    )
    meta["phaseA_final_status"] = phase_status
    return out, meta, {"x_stage1": x1, "x_stage2": x2, "rm_upper": model_inputs["row_upper"], "caps": model_inputs["caps"], "coeff": model_inputs["coeff"]}


def audit_phaseA_solution(audit_payload: Dict[str, np.ndarray], stage2_success: bool = True) -> None:
    x1 = audit_payload["x_stage1"].astype(int)
    x2 = audit_payload["x_stage2"].astype(int)
    coeff = audit_payload["coeff"]
    caps = audit_payload["caps"]
    rm_upper = audit_payload["rm_upper"]

    if stage2_success:
        assert int(x2.sum()) == int(x1.sum())
    assert np.all(x2 <= caps + 1e-9)
    assert np.all(x2 >= 0)
    assert np.allclose(x2, np.rint(x2))
    assert np.all(_rm_usage(coeff, x2.astype(np.float64)) <= rm_upper + 1e-9)


def solve_optimization(
    fg_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    rm_df: pd.DataFrame,
    mode_avail: str,
    objective: str,
    big_m_cap: int = 10**9,
    enforce_caps: bool = True,
    threads: int | None = None,
    mip_rel_gap: float | None = None,
    time_limit_sec: float | None = None,
) -> SolveOutcome:
    start = time.time()
    fg = fg_df.copy()
    fg["FG Code"] = fg["FG Code"].astype(str)
    bom = bom_df.copy()
    bom["FG Code"] = bom["FG Code"].astype(str)
    bom["RM Code"] = bom["RM Code"].astype(str)

    model_inputs = _build_model_inputs(fg, bom, cap_df, rm_df, mode_avail, enforce_caps=enforce_caps, big_m_cap=big_m_cap)
    margin_col = _fg_margin_col(fg)
    margins = pd.to_numeric(fg[margin_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if objective == "MARGIN":
        obj = margins
    elif objective == "PAIRS":
        obj = np.ones(len(model_inputs["fg_codes"]), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported objective for solve_optimization: {objective}")

    n = len(model_inputs["fg_codes"])
    mip_var_types = np.array([HighsVarType.kInteger] * n)
    lp_var_types = np.array([HighsVarType.kContinuous] * n)

    status, vals, obj_val, runtime = _solve_single_objective(
        model_inputs,
        obj,
        var_types=mip_var_types,
        sense="max",
        threads=threads,
        time_limit=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
    )
    solver_used = "highs_mip"
    used_fallback = False

    if not _status_feasible(status):
        lp_status, lp_vals, _, _ = _solve_single_objective(
            model_inputs,
            obj,
            var_types=lp_var_types,
            sense="max",
            threads=threads,
            time_limit=time_limit_sec,
            mip_rel_gap=mip_rel_gap,
        )
        base = np.maximum(np.floor(lp_vals), 0).astype(int) if _status_feasible(lp_status) else np.zeros(len(obj), dtype=int)
        target = int(model_inputs["caps"].sum()) if enforce_caps else int(base.sum() + 10_000)
        ints, greedy_meta = _greedy_fill(base, target, model_inputs["caps"], model_inputs["coeff"], model_inputs["row_upper"], obj)
        vals = ints.astype(np.float64)
        obj_val = float(np.dot(obj, vals))
        status = f"fallback_{lp_status}"
        solver_used = "lp+greedy"
        used_fallback = True
        heuristic_cutoff_hit = bool(greedy_meta["heuristic_cutoff_hit"])
        heuristic_iterations = int(greedy_meta["iterations"])
        fallback_elapsed_sec = float(greedy_meta["elapsed_sec"])
    else:
        heuristic_cutoff_hit = False
        heuristic_iterations = 0
        fallback_elapsed_sec = 0.0

    quantities = {k: int(max(0, round(v))) for k, v in zip(model_inputs["fg_codes"], vals)}
    return SolveOutcome(
        quantities,
        float(obj_val),
        status,
        "highs",
        used_fallback,
        solver_used,
        time.time() - start,
        heuristic_cutoff_hit=heuristic_cutoff_hit,
        heuristic_iterations=heuristic_iterations,
        fallback_elapsed_sec=fallback_elapsed_sec,
    )


def solve_purchase_plan_pairs_target(
    fg_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    rm_df: pd.DataFrame,
    mode_avail: str,
    target_fill_pct: float,
    threads: int | None = None,
    mip_rel_gap: float | None = None,
    time_limit_sec: float | None = None,
) -> Dict[str, object]:
    start = time.time()
    model_inputs = _build_model_inputs(fg_df, bom_df, cap_df, rm_df, mode_avail, enforce_caps=True, big_m_cap=10**9)
    fg_codes = model_inputs["fg_codes"]
    rm_codes = rm_df["RM Code"].astype(str).tolist()
    n_x = len(fg_codes)
    n_r = len(rm_codes)

    rm_rate_col = _rm_rate_col(rm_df)
    rm_rate_map = dict(zip(rm_codes, pd.to_numeric(rm_df[rm_rate_col], errors="coerce").fillna(0.0)))
    rm_rates = np.array([float(rm_rate_map[rm]) for rm in rm_codes], dtype=np.float64)

    caps = model_inputs["caps"]
    target_pairs = int(np.ceil(float(target_fill_pct) * float(caps.sum())))

    col_entries: List[List[Tuple[int, float]]] = [list(entries) for entries in model_inputs["col_entries"]]
    pair_row = n_r
    for j in range(n_x):
        col_entries[j].append((pair_row, 1.0))
    for r in range(n_r):
        col_entries.append([(r, -1.0)])

    col_cost = np.concatenate([np.zeros(n_x, dtype=np.float64), rm_rates])
    col_lower = np.zeros(n_x + n_r, dtype=np.float64)
    col_upper = np.concatenate([caps, np.full(n_r, kHighsInf, dtype=np.float64)])
    row_lower = np.concatenate([np.full(n_r, -kHighsInf, dtype=np.float64), np.array([float(target_pairs)], dtype=np.float64)])
    row_upper = np.concatenate([model_inputs["row_upper"], np.array([kHighsInf], dtype=np.float64)])

    var_types = np.array([HighsVarType.kInteger] * n_x + [HighsVarType.kContinuous] * n_r)
    mip_status, values, objective_value, runtime = _solve_highs(
        col_cost=col_cost,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        col_entries=col_entries,
        var_types=var_types,
        sense="min",
        threads=threads,
        time_limit=time_limit_sec,
        mip_rel_gap=mip_rel_gap,
    )

    status = mip_status
    lp_status = "not_run"
    method = "highs_mip"
    heuristic_cutoff_hit = False
    cutoff_reason = "none"
    fallback_iterations = 0
    fallback_elapsed_sec = 0.0
    fallback_initial_pairs = 0
    fallback_final_pairs = 0

    if not _status_feasible(mip_status):
        lp_var_types = np.array([HighsVarType.kContinuous] * (n_x + n_r))
        lp_status, lp_values, _, runtime = _solve_highs(
            col_cost=col_cost,
            col_lower=col_lower,
            col_upper=col_upper,
            row_lower=row_lower,
            row_upper=row_upper,
            col_entries=col_entries,
            var_types=lp_var_types,
            sense="min",
            threads=threads,
            time_limit=time_limit_sec,
            mip_rel_gap=mip_rel_gap,
        )

        base_x = np.maximum(np.floor(lp_values[:n_x]), 0).astype(int) if _status_feasible(lp_status) else np.zeros(n_x, dtype=int)
        coeff = model_inputs["coeff"]
        rm_upper = model_inputs["row_upper"]
        fallback_initial_pairs = int(base_x.sum())
        if int(base_x.sum()) < target_pairs:
            base_x, greedy_meta = _greedy_fill(base_x, target_pairs, caps, coeff, rm_upper, np.ones(n_x, dtype=float))
            heuristic_cutoff_hit = bool(greedy_meta["heuristic_cutoff_hit"])
            cutoff_reason = str(greedy_meta["cutoff_reason"])
            fallback_iterations = int(greedy_meta["iterations"])
            fallback_elapsed_sec = float(greedy_meta["elapsed_sec"])

        fallback_final_pairs = int(base_x.sum())

        usage = coeff @ base_x.astype(float)
        deficits = np.maximum(usage - rm_upper, 0.0)
        values = np.concatenate([base_x.astype(float), deficits])
        if fallback_final_pairs >= target_pairs:
            status = "fallback_feasible"
        elif heuristic_cutoff_hit and fallback_final_pairs > fallback_initial_pairs:
            status = "fallback_cutoff_partial"
        else:
            status = "fallback_no_progress"
        method = "fallback_deficit_buy"
        objective_value = float(np.dot(rm_rates, deficits))

    x_vals = np.maximum(np.rint(values[:n_x]), 0).astype(int)
    buy_vals = np.maximum(values[n_x:], 0.0)
    total_buy_cost = float(np.dot(rm_rates, buy_vals))

    return {
        "status": status,
        "mip_status": mip_status,
        "lp_status": lp_status,
        "solver": "highs",
        "method": method,
        "runtime_sec": time.time() - start if runtime is None else float(runtime),
        "target_fill_pct": float(target_fill_pct),
        "target_pairs": target_pairs,
        "x": {fg_codes[i]: int(x_vals[i]) for i in range(n_x)},
        "buy": {rm_codes[r]: float(buy_vals[r]) for r in range(n_r)},
        "total_buy_cost": total_buy_cost,
        "heuristic_cutoff_hit": bool(heuristic_cutoff_hit),
        "cutoff_reason": cutoff_reason,
        "fallback_iterations": int(fallback_iterations),
        "fallback_elapsed_sec": float(fallback_elapsed_sec),
        "fallback_initial_pairs": int(fallback_initial_pairs),
        "fallback_final_pairs": int(fallback_final_pairs),
    }


def solve_purchase_planner_milp(
    fg_df: pd.DataFrame,
    bom_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    rm_df: pd.DataFrame,
    mode_avail: str,
    scenario: str,
    target_value: float,
) -> Dict[str, object]:
    total_cap = float(pd.to_numeric(cap_df[_cap_col(cap_df)], errors="coerce").fillna(0.0).sum())
    scenario_key = str(scenario).upper()
    if scenario_key == "PAIRS":
        target_pairs = float(target_value)
    else:
        margins = pd.to_numeric(fg_df[_fg_margin_col(fg_df)], errors="coerce").fillna(0.0)
        plan_margin_max = float((margins.to_numpy(dtype=float) * pd.to_numeric(cap_df[_cap_col(cap_df)], errors="coerce").fillna(0.0).to_numpy(dtype=float)).sum())
        fill = 0.0 if plan_margin_max <= 0 else float(target_value) / plan_margin_max
        target_pairs = fill * total_cap
    fill_pct = 0.0 if total_cap <= 0 else float(target_pairs) / total_cap
    out = solve_purchase_plan_pairs_target(fg_df, bom_df, cap_df, rm_df, mode_avail, fill_pct)
    out["y"] = out.get("buy", {})
    out["objective_value"] = out.get("total_buy_cost", 0.0)
    return out
