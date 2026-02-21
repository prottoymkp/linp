import pandas as pd
import pytest

from app.validate import ValidationError, validate_inputs


def _valid_tables():
    return {
        "fg_master": pd.DataFrame({"FG Code": ["A"], "Dealer Price": [10], "Cost Value": [7], "Margin": [3]}),
        "bom_master": pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [2]}),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A"], "Max Plan Qty": [5]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [20], "Avail_StockPO": [20], "RM_Rate": [1.5]}),
        "tblControl_2": pd.DataFrame({"Key": ["Mode_Avail", "Objective"], "Value": ["STOCK", "PAIRS"]}),
    }


def test_validate_inputs_ok():
    validate_inputs(_valid_tables())


def test_validate_inputs_bad_control():
    t = _valid_tables()
    t["tblControl_2"] = pd.DataFrame({"Key": ["Mode_Avail", "Objective"], "Value": ["X", "Y"]})
    with pytest.raises(ValidationError):
        validate_inputs(t)


def test_validate_inputs_objective_case_insensitive_plan():
    t = _valid_tables()
    t["tblControl_2"] = pd.DataFrame({"Key": ["Mode_Avail", "Objective"], "Value": ["stock", "plan"]})
    validate_inputs(t)


def test_validate_inputs_missing_rm_rate_column():
    t = _valid_tables()
    t["tblRMAvail"] = t["tblRMAvail"].drop(columns=["RM_Rate"])

    with pytest.raises(ValidationError) as exc:
        validate_inputs(t)

    assert "tblRMAvail missing columns: RM_Rate" in str(exc.value)


def test_validate_inputs_non_numeric_rm_rate_fails():
    t = _valid_tables()
    t["tblRMAvail"] = pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [20], "Avail_StockPO": [20], "RM_Rate": [" "]})

    with pytest.raises(ValidationError) as exc:
        validate_inputs(t)

    assert "tblRMAvail.RM_Rate has non-numeric values" in str(exc.value)


def test_validate_inputs_negative_rm_rate_fails():
    t = _valid_tables()
    t["tblRMAvail"].loc[0, "RM_Rate"] = -0.1

    with pytest.raises(ValidationError) as exc:
        validate_inputs(t)

    assert "tblRMAvail.RM_Rate has negative values" in str(exc.value)


def test_validate_inputs_zero_and_positive_rm_rate_ok():
    t = _valid_tables()
    t["tblRMAvail"] = pd.DataFrame(
        {"RM Code": ["R1", "R2"], "Avail_Stock": [20, 30], "Avail_StockPO": [20, 15], "RM_Rate": [0, 2.25]}
    )
    t["bom_master"] = pd.DataFrame({"FG Code": ["A", "A"], "RM Code": ["R1", "R2"], "QtyPerPair": [2, 1]})

    validate_inputs(t)
