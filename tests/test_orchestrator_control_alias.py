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

def test_run_optimization_reads_control_keys_case_insensitively(monkeypatch):
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
                "Setting": ["mode_avail", "objective", "Threads"],
                "Value": ["STOCK", "PAIRS", "8"],
            }
        ),
    }

    _, _, meta, _, _ = run_optimization(tables)
    meta_map = dict(zip(meta["Key"], meta["Value"]))

    assert meta_map["threads"] == "8"


def test_run_optimization_accepts_rm_rate_alias_for_purchase_planning(monkeypatch):
    def fake_solve_optimization(fg, bom, cap, rm, mode_avail, objective, big_m_cap, enforce_caps, **kwargs):
        return SolveOutcome({"A": 1}, 1.0, "Optimal", "stub", False, "stub", 0.0)

    def fake_purchase_solver(**kwargs):
        assert "RM_Rate" in kwargs["rm_df"].columns
        assert kwargs["rm_df"]["RM_Rate"].iloc[0] == 2.5
        return {
            "status": "Optimal",
            "mip_status": "Optimal",
            "lp_status": "Optimal",
            "method": "highs_mip",
            "x": {"A": 1},
            "buy": {"R1": 0.0},
            "total_buy_cost": 0.0,
            "heuristic_cutoff_hit": False,
            "cutoff_reason": "none",
            "fallback_iterations": 0,
            "fallback_elapsed_sec": 0.0,
        }

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve_optimization)
    monkeypatch.setattr("app.orchestrator.solve_purchase_plan_pairs_target", fake_purchase_solver)

    tables = {
        "fg_master": pd.DataFrame({"FG Code": ["A"], "Margin": [3]}),
        "bom_master": pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [1]}),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A"], "Max Plan Qty": [1]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [1], "Avail_StockPO": [1], "RM Rate": [2.5]}),
        "tblControl_2": pd.DataFrame({"Setting": ["Mode_Avail", "Objective"], "Value": ["STOCK", "PAIRS"]}),
    }

    _, _, meta, purchase_summary, _ = run_optimization(tables, run_purchase_planner=True, purchase_target_fill_pcts="100")
    meta_map = dict(zip(meta["Key"], meta["Value"]))

    assert meta_map["purchase_plan_status"] == "ran"
    assert purchase_summary["Status"].tolist() == ["Optimal"]


def test_run_optimization_caps_default_threads_for_safer_concurrency(monkeypatch):
    def fake_solve_optimization(fg, bom, cap, rm, mode_avail, objective, big_m_cap, enforce_caps, **kwargs):
        return SolveOutcome({"A": 1}, 1.0, "Optimal", "stub", False, "stub", 0.0)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve_optimization)
    monkeypatch.setattr("app.orchestrator.os.cpu_count", lambda: 16)

    tables = {
        "fg_master": pd.DataFrame({"FG Code": ["A"], "Margin": [3]}),
        "bom_master": pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [1]}),
        "tblFGPlanCap": pd.DataFrame({"FG Code": ["A"], "Max Plan Qty": [1]}),
        "tblRMAvail": pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [1], "Avail_StockPO": [1]}),
        "tblControl_2": pd.DataFrame({"Setting": ["Mode_Avail", "Objective"], "Value": ["STOCK", "PAIRS"]}),
    }

    _, _, meta, _, _ = run_optimization(tables)
    meta_map = dict(zip(meta["Key"], meta["Value"]))

    assert meta_map["threads"] == "4"
