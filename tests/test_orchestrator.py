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
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [rm_avail], "Avail_StockPO": [rm_avail], "RM_Rate": [1.0]}),
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
    assert len(purchase_summary) == 4
    assert set(purchase_summary["Status"]) == {"not_run"}

    meta_map = dict(zip(meta["Key"], meta["Value"]))
    assert float(meta_map["TotalCapPairs"]) == float(fg["Plan Cap"].sum())
    assert float(meta_map["AchievedPairs"]) == float(fg["Opt Qty Total"].sum())
    assert float(meta_map["PlanMarginMax"]) == float((fg["Plan Cap"] * fg["Unit Margin"]).sum())
    assert float(meta_map["AchievedMargin"]) == float(fg["Total Margin"].sum())


def test_phase_b_runs_when_caps_met():
    fg, _, meta, purchase_summary, _ = run_optimization(_tables(rm_avail=12, cap=4))
    assert fg["Opt Qty Phase B"].sum() >= 0
    assert meta.loc[meta["Key"] == "all_caps_hit", "Value"].iloc[0] == "True"
    assert set(purchase_summary["Status"]) == {"not_run"}


def test_purchase_planner_summary_when_enabled():
    fg, rm, meta, purchase_summary, purchase_detail = run_optimization(
        _tables(rm_avail=3, cap=4),
        run_purchase_planner=True,
    )

    assert not fg.empty
    assert not rm.empty
    assert not meta.empty
    assert not purchase_summary.empty
    assert "Status" in purchase_summary.columns
    assert "BuyCost" in purchase_detail.columns


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


def test_purchase_planner_preserves_fallback_status_fields(monkeypatch):
    def fake_purchase_solver(**kwargs):
        fg_codes = kwargs["cap_df"]["FG Code"].astype(str).tolist()
        rm_codes = kwargs["rm_df"]["RM Code"].astype(str).tolist()
        return {
            "status": "fallback_cutoff_partial",
            "mip_status": "Time limit reached",
            "lp_status": "Infeasible",
            "method": "fallback_deficit_buy",
            "x": {code: 1 for code in fg_codes},
            "buy": {code: 1.0 for code in rm_codes},
            "total_buy_cost": 1.0,
            "heuristic_cutoff_hit": True,
            "cutoff_reason": "max_iterations",
            "fallback_iterations": 10,
            "fallback_elapsed_sec": 0.5,
        }

    monkeypatch.setattr("app.orchestrator.solve_purchase_plan_pairs_target", fake_purchase_solver)

    _, _, _, purchase_summary, _ = run_optimization(_tables(rm_avail=1, cap=4), run_purchase_planner=True)

    assert set(purchase_summary["Status"]) == {"fallback_cutoff_partial"}
    assert set(purchase_summary["MIPStatus"]) == {"Time limit reached"}
    assert set(purchase_summary["LPStatus"]) == {"Infeasible"}
    assert purchase_summary["HeuristicCutoffHit"].all()
    assert set(purchase_summary["CutoffReason"]) == {"max_iterations"}
