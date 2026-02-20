"""Schema and business-rule validation with aggregated fail-fast errors."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from app.config import (
    BOM_DATASET,
    CAP_DATASET,
    CONTROL_DATASET,
    CONTROL_KEYS,
    FG_DATASET,
    REQUIRED_COLUMNS,
    REQUIRED_TABLES,
    RM_DATASET,
)
from app.types import RunConfig


class ValidationError(Exception):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("Validation failed:\n" + "\n".join(f"- {e}" for e in errors))


def _check_numeric(df: pd.DataFrame, column: str, errors: List[str], *, non_negative: bool = True, strictly_positive: bool = False, integer_like: bool = False, label: str = "") -> None:
    if column not in df.columns:
        return
    series = pd.to_numeric(df[column], errors="coerce")
    if series.isna().any() or (~np.isfinite(series)).any():
        errors.append(f"{label}{column} contains null/non-finite values.")
        return
    if non_negative and (series < 0).any():
        errors.append(f"{label}{column} must be non-negative.")
    if strictly_positive and (series <= 0).any():
        errors.append(f"{label}{column} must be > 0.")
    if integer_like and (~np.isclose(series, np.floor(series))).any():
        errors.append(f"{label}{column} must be integer-like values.")


def validate_inputs(data: Dict[str, pd.DataFrame]) -> RunConfig:
    errors: List[str] = []

    missing_tables = sorted(REQUIRED_TABLES - set(data.keys()))
    if missing_tables:
        errors.append(f"Missing required tables: {', '.join(missing_tables)}")

    for dataset, required_cols in REQUIRED_COLUMNS.items():
        if dataset not in data:
            continue
        columns = list(data[dataset].columns)
        missing = [col for col in required_cols if col not in columns]
        extra = [col for col in columns if col not in required_cols]
        if missing:
            errors.append(f"{dataset} missing columns: {', '.join(missing)}")
        if extra:
            errors.append(f"{dataset} has unexpected columns: {', '.join(extra)}")

    if FG_DATASET in data and data[FG_DATASET]["FG Code"].duplicated().any():
        errors.append("FG master must have unique FG Code values.")
    if RM_DATASET in data and data[RM_DATASET]["RM Code"].duplicated().any():
        errors.append("RM availability must have unique RM Code values.")

    if FG_DATASET in data:
        _check_numeric(data[FG_DATASET], "Unit Margin", errors, non_negative=False, label="fg_master.")
    if BOM_DATASET in data:
        _check_numeric(data[BOM_DATASET], "QtyPerPair", errors, strictly_positive=True, label="bom_master.")
    if CAP_DATASET in data:
        _check_numeric(data[CAP_DATASET], "Plan Cap", errors, integer_like=True, label="fg_plan_cap.")
    if RM_DATASET in data:
        _check_numeric(data[RM_DATASET], "Avail_Stock", errors, integer_like=True, label="rm_avail.")
        _check_numeric(data[RM_DATASET], "Avail_StockPO", errors, integer_like=True, label="rm_avail.")

    if FG_DATASET in data and CAP_DATASET in data:
        missing_fg = sorted(set(data[CAP_DATASET]["FG Code"]) - set(data[FG_DATASET]["FG Code"]))
        if missing_fg:
            errors.append(f"FG codes in cap not found in FG master: {', '.join(map(str, missing_fg))}")
    if FG_DATASET in data and BOM_DATASET in data:
        missing_fg = sorted(set(data[BOM_DATASET]["FG Code"]) - set(data[FG_DATASET]["FG Code"]))
        if missing_fg:
            errors.append(f"FG codes in BOM not found in FG master: {', '.join(map(str, missing_fg))}")
    if RM_DATASET in data and BOM_DATASET in data:
        missing_rm = sorted(set(data[BOM_DATASET]["RM Code"]) - set(data[RM_DATASET]["RM Code"]))
        if missing_rm:
            errors.append(f"RM codes in BOM not found in RM availability: {', '.join(map(str, missing_rm))}")

    run_config = RunConfig(mode_avail="", objective="", phase_b_upper_bound=1_000_000)
    if CONTROL_DATASET in data:
        ctl = data[CONTROL_DATASET].copy()
        ctl_map = {
            str(row["Control Key"]).strip(): str(row["Control Value"]).strip().upper()
            for _, row in ctl.iterrows()
        }
        for key, allowed in CONTROL_KEYS.items():
            value = ctl_map.get(key)
            if value not in allowed:
                errors.append(f"Control {key} must be one of {sorted(allowed)}, got {value!r}")
        phase_b_ub = ctl_map.get("PhaseB_UpperBound", "1000000")
        try:
            phase_b_upper_bound = int(float(phase_b_ub))
            if phase_b_upper_bound <= 0:
                raise ValueError
        except ValueError:
            errors.append("Control PhaseB_UpperBound must be a positive integer if provided.")
            phase_b_upper_bound = 1_000_000
        run_config = RunConfig(
            mode_avail=ctl_map.get("Mode_Avail", ""),
            objective=ctl_map.get("Objective", ""),
            phase_b_upper_bound=phase_b_upper_bound,
        )

    if errors:
        raise ValidationError(errors)
    return run_config
