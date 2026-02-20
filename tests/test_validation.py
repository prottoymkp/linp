import pandas as pd
import pytest

from app.config import BOM_DATASET, CAP_DATASET, CONTROL_DATASET, FG_DATASET, RM_DATASET
from app.validate import ValidationError, validate_inputs


def valid_payload():
    return {
        FG_DATASET: pd.DataFrame({"FG Code": ["FG1"], "Unit Margin": [10]}),
        BOM_DATASET: pd.DataFrame({"FG Code": ["FG1"], "RM Code": ["RM1"], "QtyPerPair": [1]}),
        CAP_DATASET: pd.DataFrame({"FG Code": ["FG1"], "Plan Cap": [5]}),
        RM_DATASET: pd.DataFrame({"RM Code": ["RM1"], "Avail_Stock": [10], "Avail_StockPO": [20]}),
        CONTROL_DATASET: pd.DataFrame(
            {
                "Control Key": ["Mode_Avail", "Objective"],
                "Control Value": ["STOCK", "MARGIN"],
            }
        ),
    }


def test_missing_table_and_columns_fail():
    payload = valid_payload()
    payload.pop(BOM_DATASET)
    payload[FG_DATASET] = pd.DataFrame({"FG Code": ["FG1"], "Wrong": [1]})

    with pytest.raises(ValidationError) as exc:
        validate_inputs(payload)

    msg = str(exc.value)
    assert "Missing required tables" in msg
    assert "missing columns" in msg


def test_invalid_control_values_fail():
    payload = valid_payload()
    payload[CONTROL_DATASET] = pd.DataFrame(
        {
            "Control Key": ["Mode_Avail", "Objective"],
            "Control Value": ["BAD", "WRONG"],
        }
    )

    with pytest.raises(ValidationError) as exc:
        validate_inputs(payload)

    msg = str(exc.value)
    assert "Control Mode_Avail" in msg
    assert "Control Objective" in msg


def test_control_setting_value_columns_pass():
    payload = valid_payload()
    payload[CONTROL_DATASET] = pd.DataFrame(
        {
            "Setting": ["Horizon_Start", "Horizon_End", "Mode_Avail", "Objective"],
            "Value": ["01-Dec-25", "31-Dec-25", "STOCK", "MARGIN"],
        }
    )

    validate_inputs(payload)
