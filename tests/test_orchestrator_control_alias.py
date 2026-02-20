import pandas as pd

from app.orchestrator import run_optimization


def test_run_optimization_accepts_setting_value_control_columns():
    tables = {
        "fg_master": pd.DataFrame({"FG Code": ["A"], "Margin": [3]}),
        "bom_master": pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [1]}),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A"], "Max Plan Qty": [1]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [1], "Avail_StockPO": [1]}),
        "tblControl_2": pd.DataFrame(
            {
                "Setting": ["Horizon_Start", "Horizon_End", "Mode_Avail", "Objective"],
                "Value": ["01-Dec-25", "31-Dec-25", "STOCK", "PAIRS"],
            }
        ),
    }

    fg, rm, meta = run_optimization(tables)

    assert not fg.empty
    assert not rm.empty
    assert not meta.empty
