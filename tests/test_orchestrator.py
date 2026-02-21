import pandas as pd
import pytest

from app.model import SolveOutcome
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


@pytest.fixture(autouse=True)
def _stub_solve_optimization(monkeypatch):
    def fake_solve_optimization(fg, bom, cap, rm, mode_avail, objective, big_m_cap, enforce_caps):
        cap_col = "Max Plan Qty" if "Max Plan Qty" in cap.columns else "Plan Cap"
        caps = dict(zip(cap["FG Code"].astype(str), pd.to_numeric(cap[cap_col], errors="coerce").fillna(0).astype(int)))
        avail_col = "Avail_Stock" if mode_avail == "STOCK" else "Avail_StockPO"
        avail = float(pd.to_numeric(rm[avail_col], errors="coerce").fillna(0).sum())

        if enforce_caps:
            if avail >= sum(caps.values()):
                quantities = {code: qty for code, qty in caps.items()}
            else:
                quantities = {code: 0 for code in caps}
                if caps:
                    first_code = next(iter(caps))
                    quantities[first_code] = min(caps[first_code], int(avail))
            status = "Optimal"
        else:
            quantities = {code: 0 for code in caps}
            if avail >= 1 and caps:
                first_code = next(iter(caps))
                quantities[first_code] = 1
            status = "Optimal"

        return SolveOutcome(quantities, float(sum(quantities.values())), status, "stub", False, "stub", 0.0)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve_optimization)


def test_phase_b_zero_when_caps_not_met():
    fg, _, meta, purchase_summary, _ = run_optimization(_tables(rm_avail=3, cap=4))
    assert fg["Opt Qty Phase B"].sum() == 0
    assert "Fill_FG" in fg.columns
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "False"
    assert len(purchase_summary) == 1
    assert purchase_summary.loc[0, "Status"] == "skipped"

    meta_map = dict(zip(meta["Key"], meta["Value"]))
    assert float(meta_map["TotalCapPairs"]) == float(fg["Plan Cap"].sum())
    assert float(meta_map["AchievedPairs"]) == float(fg["Opt Qty Total"].sum())
    assert float(meta_map["PlanMarginMax"]) == float((fg["Plan Cap"] * fg["Unit Margin"]).sum())
    assert float(meta_map["AchievedMargin"]) == float(fg["Total Margin"].sum())


def test_phase_b_runs_when_caps_met():
    fg, _, meta, purchase_summary, _ = run_optimization(_tables(rm_avail=12, cap=4))
    assert fg["Opt Qty Phase B"].sum() >= 0
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "True"
    assert purchase_summary.loc[0, "Status"] in {"Optimal", "Feasible", "fallback_Optimal", "fallback_Feasible"}


def test_purchase_planner_summary_when_enabled():
    fg, rm, meta, purchase_summary, purchase_detail = run_optimization(_tables(rm_avail=3, cap=4), run_purchase_planner=True)

    assert not fg.empty
    assert not rm.empty
    assert not meta.empty
    assert purchase_summary is not None
    assert purchase_detail.empty
    assert len(purchase_summary) == 1
    assert set(["TargetMetric", "AchievedPairs", "AchievedMargin", "Status"]).issubset(purchase_summary.columns)


def test_fill_fg_guard_when_plan_cap_zero():
    fg, _, _, _, _ = run_optimization(_tables(rm_avail=5, cap=0))
    assert (fg["Plan Cap"] == 0).all()
    assert (fg["Fill_FG"] == 0).all()


def test_run_optimization_returns_canonical_five_part_contract():
    result = run_optimization(_tables(rm_avail=5, cap=4))

    assert isinstance(result, tuple)
    assert len(result) == 5
    fg, rm, meta, purchase_summary, purchase_detail = result
    assert isinstance(fg, pd.DataFrame)
    assert isinstance(rm, pd.DataFrame)
    assert isinstance(meta, pd.DataFrame)
    assert isinstance(purchase_summary, pd.DataFrame)
    assert isinstance(purchase_detail, pd.DataFrame)
