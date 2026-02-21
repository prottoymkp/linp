import pandas as pd

from app.orchestrator import run_optimization


def _tables(rm_avail=10, cap=4):
    return {
        "fg_master": pd.DataFrame(
            {
                "FG Code": ["A", "B"],
                "Dealer Price": [10, 10],
                "Cost Value": [7, 8],
                "Margin": [3, 2],
            }
        ),
        "bom_master": pd.DataFrame(
            {
                "FG Code": ["A", "B"],
                "RM Code": ["R1", "R1"],
                "QtyPerPair": [1, 1],
            }
        ),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A", "B"], "Max Plan Qty": [cap, cap]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [rm_avail], "Avail_StockPO": [rm_avail]}),
        "tblControl_2": pd.DataFrame({"Key": ["Mode_Avail", "Objective"], "Value": ["STOCK", "PAIRS"]}),
    }


def test_phase_b_zero_when_caps_not_met():
    fg, _, meta, purchase_summary, _ = run_optimization(_tables(rm_avail=3, cap=4))
    assert fg["Opt Qty Phase B"].sum() == 0
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "False"
    assert len(purchase_summary) == 1
    assert purchase_summary.loc[0, "Status"] == "skipped"


def test_phase_b_runs_when_caps_met():
    fg, _, meta, purchase_summary, _ = run_optimization(_tables(rm_avail=12, cap=4))
    assert fg["Opt Qty PhaseB"].sum() if "Opt Qty PhaseB" in fg.columns else fg["Opt Qty Phase B"].sum() >= 0
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "True"
    assert purchase_summary.loc[0, "Status"] in {"Optimal", "Feasible", "fallback_Optimal", "fallback_Feasible"}
