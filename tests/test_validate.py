import pandas as pd
import pytest

from app.validate import ValidationError, validate_inputs


def _valid_tables():
    return {
        "fg_master": pd.DataFrame({"FG Code": ["A"], "Dealer Price": [10], "Cost Value": [7], "Margin": [3]}),
        "bom_master": pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [2]}),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A"], "Max Plan Qty": [5]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [20], "Avail_StockPO": [20]}),
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
