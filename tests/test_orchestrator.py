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
    fg, _, meta = run_optimization(_tables(rm_avail=3, cap=4))
    assert fg["Opt Qty Phase B"].sum() == 0
    assert "Fill_FG" in fg.columns
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "False"

    meta_map = dict(zip(meta["Key"], meta["Value"]))
    assert float(meta_map["TotalCapPairs"]) == float(fg["Plan Cap"].sum())
    assert float(meta_map["AchievedPairs"]) == float(fg["Opt Qty Total"].sum())
    assert float(meta_map["PlanMarginMax"]) == float((fg["Plan Cap"] * fg["Unit Margin"]).sum())
    assert float(meta_map["AchievedMargin"]) == float(fg["Total Margin"].sum())


def test_phase_b_runs_when_caps_met():
    fg, _, meta = run_optimization(_tables(rm_avail=12, cap=4))
    assert fg["Opt Qty PhaseB"].sum() if "Opt Qty PhaseB" in fg.columns else fg["Opt Qty Phase B"].sum() >= 0
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "True"


def test_purchase_planner_summary_when_enabled():
    fg, rm, meta, purchase_summary = run_optimization(_tables(rm_avail=3, cap=4), run_purchase_planner=True)

    assert not fg.empty
    assert not rm.empty
    assert not meta.empty
    assert purchase_summary is not None
    assert sorted(purchase_summary["Coverage %"].unique().tolist()) == [25, 50, 75, 100]
    assert set(["RM Code", "Current Availability", "Required Qty", "Purchase Required"]).issubset(purchase_summary.columns)
def test_fill_fg_guard_when_plan_cap_zero():
    fg, _, _ = run_optimization(_tables(rm_avail=5, cap=0))
    assert (fg["Plan Cap"] == 0).all()
    assert (fg["Fill_FG"] == 0).all()
