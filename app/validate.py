from typing import Dict, List, Tuple

import pandas as pd

from .config import (
    BOM_DATASET,
    CAP_DATASET,
    CONTROL_DATASET,
    CONTROL_MODE_VALUES,
    CONTROL_OBJECTIVE_VALUES,
    FG_DATASET,
    RM_DATASET,
)


class ValidationError(Exception):
    pass


def _control_cols(df: pd.DataFrame) -> Tuple[str, str]:
    if {"Key", "Value"}.issubset(df.columns):
        return "Key", "Value"
    if {"Control Key", "Control Value"}.issubset(df.columns):
        return "Control Key", "Control Value"
    raise ValidationError("Control table must have columns ['Key','Value'] or ['Control Key','Control Value']")


def _fg_margin_col(df: pd.DataFrame) -> str:
    if "Margin" in df.columns:
        return "Margin"
    if "Unit Margin" in df.columns:
        return "Unit Margin"
    raise ValidationError("fg_master must include either 'Margin' or 'Unit Margin'")


def _cap_col(df: pd.DataFrame) -> str:
    if "Max Plan Qty" in df.columns:
        return "Max Plan Qty"
    if "Plan Cap" in df.columns:
        return "Plan Cap"
    raise ValidationError("tblFGPlanCap must include either 'Max Plan Qty' or 'Plan Cap'")


def validate_inputs(tables: Dict[str, pd.DataFrame]) -> None:
    errors: List[str] = []

    required = {FG_DATASET, BOM_DATASET, CAP_DATASET, RM_DATASET, CONTROL_DATASET}
    missing = sorted(list(required - set(tables.keys())))
    if missing:
        errors.append(f"Missing required tables: {missing}")

    if errors:
        raise ValidationError("\n".join(errors))

    fg = tables[FG_DATASET]
    bom = tables[BOM_DATASET]
    cap = tables[CAP_DATASET]
    rm = tables[RM_DATASET]
    ctrl = tables[CONTROL_DATASET]

    for col in ["FG Code"]:
        if col not in fg.columns:
            errors.append(f"fg_master missing columns: {col}")
    for col in ["FG Code", "RM Code", "QtyPerPair"]:
        if col not in bom.columns:
            errors.append(f"bom_master missing columns: {col}")
    for col in ["FG Code"]:
        if col not in cap.columns:
            errors.append(f"tblFGPlanCap missing columns: {col}")
    for col in ["RM Code", "Avail_Stock", "Avail_StockPO"]:
        if col not in rm.columns:
            errors.append(f"tblRMAvail missing columns: {col}")

    try:
        _ = _fg_margin_col(fg)
    except ValidationError as e:
        errors.append(str(e))
    try:
        _ = _cap_col(cap)
    except ValidationError as e:
        errors.append(str(e))

    if errors:
        raise ValidationError("\n".join(errors))

    if fg["FG Code"].duplicated().any():
        errors.append("Duplicate FG Code in fg_master")
    if rm["RM Code"].duplicated().any():
        errors.append("Duplicate RM Code in tblRMAvail")

    fg_margin_col = _fg_margin_col(fg)
    cap_col = _cap_col(cap)
    numeric_checks = [
        (fg, [fg_margin_col], "fg_master"),
        (bom, ["QtyPerPair"], "bom_master"),
        (cap, [cap_col], "tblFGPlanCap"),
        (rm, ["Avail_Stock", "Avail_StockPO"], "tblRMAvail"),
    ]
    for df, cols, name in numeric_checks:
        for col in cols:
            ser = pd.to_numeric(df[col], errors="coerce")
            if ser.isna().any():
                errors.append(f"{name}.{col} has non-numeric values")
            if (ser < 0).any():
                errors.append(f"{name}.{col} has negative values")
    if (pd.to_numeric(bom["QtyPerPair"], errors="coerce") <= 0).any():
        errors.append("bom_master.QtyPerPair must be > 0")

    fg_codes = set(fg["FG Code"].astype(str))
    if not set(bom["FG Code"].astype(str)).issubset(fg_codes):
        errors.append("All BOM FG codes must exist in fg_master")
    if not set(cap["FG Code"].astype(str)).issubset(fg_codes):
        errors.append("All Cap FG codes must exist in fg_master")

    rm_codes = set(rm["RM Code"].astype(str))
    if not set(bom["RM Code"].astype(str)).issubset(rm_codes):
        errors.append("All BOM RM codes must exist in tblRMAvail")

    key_col, val_col = _control_cols(ctrl)
    c_map = dict(zip(ctrl[key_col].astype(str), ctrl[val_col].astype(str)))
    if c_map.get("Mode_Avail") not in CONTROL_MODE_VALUES:
        errors.append("Control Mode_Avail invalid")
    if c_map.get("Objective") not in CONTROL_OBJECTIVE_VALUES:
        errors.append("Control Objective invalid")

    if errors:
        raise ValidationError("\n".join(errors))
