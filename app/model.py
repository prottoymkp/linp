from __future__ import annotations

import time
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
    var_types: np.ndarray | None = None,
    sense: ObjSense | str = ObjSense.kMaximize,
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
    if time_limit is not None:
        highs.setOptionValue("time_limit", float(time_limit))
    if mip_rel_gap is not None:
        highs.setOptionValue("mip_rel_gap", float(mip_rel_gap))
    highs.passModel(model)
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

    if var_types is None:
        var_types = np.array([HighsVarType.kInteger] * len(col_cost))
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
) -> np.ndarray:
    x = base_x.astype(int).copy()
    usage = _rm_usage(coeff, x.astype(np.float64))
    order = np.argsort(-scores)

    while int(x.sum()) < target_total:
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
        if not moved:
            break
    return x


def _fallback_stage2_reallocate(x_stage1: np.ndarray, margins: np.ndarray, caps: np.ndarray, coeff: np.ndarray, rm_upper: np.ndarray) -> np.ndarray:
    x = x_stage1.astype(int).copy()
    usage = _rm_usage(coeff, x.astype(np.float64))
    residual = rm_upper - usage

    low_to_high = np.argsort(margins)
    high_to_low = np.argsort(-margins)

    improved = True
    while improved:
        improved = False
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
                    break
            if improved:
                break
    return x


def _solve_single_objective(
    model_inputs: Dict[str, object],
    objective: np.ndarray,
    var_types: np.ndarray | None = None,
    sense: ObjSense | str = ObjSense.kMaximize,
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
    }

    # Stage 1: maximize pairs
    mip_var_types = np.array([HighsVarType.kInteger] * n)
    lp_var_types = np.array([HighsVarType.kContinuous] * n)

    s1_status, s1_vals, _, s1_runtime = _solve_single_objective(
        model_inputs,
        np.ones(n, dtype=np.float64),
        var_types=mip_var_types,
        sense="max",
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
        )
        base = np.maximum(np.floor(lp_vals), 0).astype(int) if _status_feasible(lp_status) else np.zeros(n, dtype=int)
        x1 = _greedy_fill(base, int(base.sum()) + int(model_inputs["caps"].sum()), model_inputs["caps"], model_inputs["coeff"], model_inputs["row_upper"], np.ones(n, dtype=np.float64))
        meta["stage1_status"] = f"fallback_{lp_status}"
        meta["stage1_solver_used"] = "lp+greedy"
        meta["method"] = "lex_lp+greedy"

    p_star = int(x1.sum())
    meta["P_star"] = p_star

    if p_star <= 0:
        quantities = {code: 0 for code in fg_codes}
        out = SolveOutcome(quantities, 0.0, str(meta["stage1_status"]), "highs", True, str(meta["method"]), float(meta["stage1_runtime"]))
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
            x2_alt = _fallback_stage2_reallocate(x1, margins, model_inputs["caps"], model_inputs["coeff"], model_inputs["row_upper"])
            if int(x2_alt.sum()) == p_star:
                x2 = x2_alt
                meta["stage2_status"] = "fallback_reallocated"
                meta["stage2_solver_used"] = "greedy_reallocate"
            else:
                meta["stage2_status"] = "fallback_no_lex_stage2"

    quantities = {code: int(x2[i]) for i, code in enumerate(fg_codes)}
    out = SolveOutcome(quantities, float(s2_obj), phase_status, "highs", meta["stage1_solver_used"] != "highs_mip", str(meta["method"]), s1_runtime + s2_runtime)
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
    )
    solver_used = "highs_mip"
    used_fallback = False

    if not _status_feasible(status):
        lp_status, lp_vals, _, _ = _solve_single_objective(
            model_inputs,
            obj,
            var_types=lp_var_types,
            sense="max",
        )
        base = np.maximum(np.floor(lp_vals), 0).astype(int) if _status_feasible(lp_status) else np.zeros(len(obj), dtype=int)
        target = int(model_inputs["caps"].sum()) if enforce_caps else int(base.sum() + 10_000)
        ints = _greedy_fill(base, target, model_inputs["caps"], model_inputs["coeff"], model_inputs["row_upper"], obj)
        vals = ints.astype(np.float64)
        obj_val = float(np.dot(obj, vals))
        status = f"fallback_{lp_status}"
        solver_used = "lp+greedy"
        used_fallback = True

    quantities = {k: int(max(0, round(v))) for k, v in zip(model_inputs["fg_codes"], vals)}
    return SolveOutcome(quantities, float(obj_val), status, "highs", used_fallback, solver_used, time.time() - start)
