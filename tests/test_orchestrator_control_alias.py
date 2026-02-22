import pandas as pd

from app.model import SolveOutcome
from app.orchestrator import run_optimization


def test_run_optimization_accepts_setting_value_control_columns(monkeypatch):
    def fake_solve_optimization(fg, bom, cap, rm, mode_avail, objective, big_m_cap, enforce_caps, **kwargs):
        return SolveOutcome({"A": 1}, 1.0, "Optimal", "stub", False, "stub", 0.0)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve_optimization)

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

    fg, rm, meta, purchase_summary, purchase_detail = run_optimization(tables)

    assert not fg.empty
    assert not rm.empty
    assert not meta.empty
    assert len(purchase_summary) == 4
    assert purchase_detail.empty
    assert purchase_summary.loc[0, "Status"] in {"not_run", "skipped", "Optimal", "Feasible", "fallback_feasible", "fallback_cutoff_partial", "fallback_no_progress"}
    assert purchase_summary.loc[0, "Method"]


def test_run_optimization_emits_solver_controls_in_metadata(monkeypatch):
    def fake_solve_optimization(fg, bom, cap, rm, mode_avail, objective, big_m_cap, enforce_caps, **kwargs):
        return SolveOutcome({"A": 1}, 1.0, "Optimal", "stub", False, "stub", 0.0)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve_optimization)

    tables = {
        "fg_master": pd.DataFrame({"FG Code": ["A"], "Margin": [3]}),
        "bom_master": pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [1]}),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A"], "Max Plan Qty": [1]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [1], "Avail_StockPO": [1]}),
        "tblControl_2": pd.DataFrame(
            {
                "Setting": ["Mode_Avail", "Objective", "threads", "mip_rel_gap", "time_limit_sec"],
                "Value": ["STOCK", "PAIRS", "6", "0.03", "20"],
            }
        ),
    }

    _, _, meta, _, _ = run_optimization(tables)
    meta_map = dict(zip(meta["Key"], meta["Value"]))

    assert meta_map["threads"] == "6"
    assert meta_map["mip_rel_gap"] == "0.03"
    assert meta_map["time_limit_sec"] == "20.0"
