import numpy as np
import pandas as pd

from app.config import BOM_DATASET, CAP_DATASET, FG_DATASET, RM_DATASET
from app.orchestrator import run_two_phase
from app.types import RunConfig, SolveOutcome


def build_data():
    return {
        FG_DATASET: pd.DataFrame({"FG Code": ["FG1", "FG2"], "Unit Margin": [10, 5]}),
        BOM_DATASET: pd.DataFrame(
            {"FG Code": ["FG1", "FG2"], "RM Code": ["RM1", "RM1"], "QtyPerPair": [1, 1]}
        ),
        CAP_DATASET: pd.DataFrame({"FG Code": ["FG1", "FG2"], "Plan Cap": [2, 2]}),
        RM_DATASET: pd.DataFrame({"RM Code": ["RM1"], "Avail_Stock": [10], "Avail_StockPO": [10]}),
    }


def _audit_stub(payload, stage2_success=True):
    assert isinstance(payload["x_stage1"], np.ndarray)


def test_phase_b_zero_when_any_cap_unmet(monkeypatch):
    calls = {"n": 0}

    def fake_phase_a(*args, **kwargs):
        return (
            SolveOutcome({"FG1": 2, "FG2": 1}, 25.0, "ok", "mock", False, "lex_mip", 0.01),
            {"stage1_status": "Optimal", "stage2_status": "Optimal", "P_star": 3},
            {"x_stage1": np.array([2, 1]), "x_stage2": np.array([2, 1]), "coeff": np.ones((1, 2)), "caps": np.array([2, 2]), "rm_upper": np.array([10])},
        )

    def fake_solve(*args, **kwargs):
        calls["n"] += 1
        return SolveOutcome({"FG1": 1, "FG2": 1}, 15.0, "ok", "mock", False, "mip", 0.01)

    monkeypatch.setattr("app.orchestrator.solve_phaseA_lexicographic", fake_phase_a)
    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve)
    monkeypatch.setattr("app.orchestrator.audit_phaseA_solution", _audit_stub)
    out = run_two_phase(build_data(), RunConfig("STOCK", "PLAN", 100))

    assert out.fg_result["Opt Qty Phase B"].sum() == 0
    assert "Fill_FG" in out.fg_result.columns
    assert out.run_meta.loc[0, "phase_b_executed"] == False
    assert out.run_meta.loc[0, "TotalCapPairs"] == 4.0
    assert out.run_meta.loc[0, "AchievedPairs"] == 3.0
    assert out.run_meta.loc[0, "OverallFillPairs"] == 0.75
    assert out.run_meta.loc[0, "PlanMarginMax"] == 30.0
    assert out.run_meta.loc[0, "AchievedMargin"] == 25.0
    assert out.run_meta.loc[0, "AchievedMarginAtPairFill"] == 22.5
    assert out.run_meta.loc[0, "MarginFillAtPairFill"] == 25.0 / 22.5
    assert calls["n"] == 0


def test_phase_b_enabled_when_all_caps_met(monkeypatch):
    calls = {"n": 0}

    def fake_phase_a(*args, **kwargs):
        return (
            SolveOutcome({"FG1": 2, "FG2": 2}, 30.0, "ok", "mock", False, "lex_mip", 0.01),
            {"stage1_status": "Optimal", "stage2_status": "Optimal", "P_star": 4},
            {"x_stage1": np.array([2, 2]), "x_stage2": np.array([2, 2]), "coeff": np.ones((1, 2)), "caps": np.array([2, 2]), "rm_upper": np.array([10])},
        )

    def fake_solve(*args, **kwargs):
        calls["n"] += 1
        return SolveOutcome({"FG1": 1, "FG2": 0}, 10.0, "ok", "mock", False, "mip", 0.01)

    monkeypatch.setattr("app.orchestrator.solve_phaseA_lexicographic", fake_phase_a)
    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve)
    monkeypatch.setattr("app.orchestrator.audit_phaseA_solution", _audit_stub)
    out = run_two_phase(build_data(), RunConfig("STOCK", "PLAN", 100))

    assert out.fg_result["Opt Qty Phase B"].sum() == 1
    assert out.run_meta.loc[0, "phase_b_executed"] == True
    assert out.run_meta.loc[0, "TotalCapPairs"] == 4.0
    assert out.run_meta.loc[0, "AchievedPairs"] == 5.0
    assert out.run_meta.loc[0, "OverallFillPairs"] == 1.25
    assert out.run_meta.loc[0, "PlanMarginMax"] == 30.0
    assert out.run_meta.loc[0, "AchievedMargin"] == 40.0
    assert out.run_meta.loc[0, "AchievedMarginAtPairFill"] == 37.5
    assert out.run_meta.loc[0, "MarginFillAtPairFill"] == 40.0 / 37.5
    assert calls["n"] == 1


def test_plan_objective_calls_lexicographic_solver_once(monkeypatch):
    calls = {"lex": 0}

    def fake_phase_a(*args, **kwargs):
        calls["lex"] += 1
        return (
            SolveOutcome({"FG1": 2, "FG2": 1}, 25.0, "ok", "lex_once", False, "lex_mip", 1.23),
            {
                "method": "lex_once",
                "stage1_status": "Optimal",
                "stage1_solver_used": "lex_mip",
                "stage1_runtime": 1.23,
                "P_star": 3,
                "stage2_status": "Optimal",
                "stage2_solver_used": "lex_lp",
                "stage2_runtime": 0.11,
                "phaseA_final_status": "ok",
                "heuristic_cutoff_hit": False,
                "heuristic_iterations": 0,
                "fallback_passes": 0,
                "fallback_swaps": 0,
                "fallback_elapsed_sec": 0.0,
                "cutoff_reason": "none",
            },
            {"x_stage1": np.array([2, 1]), "x_stage2": np.array([2, 1]), "coeff": np.ones((1, 2)), "caps": np.array([2, 2]), "rm_upper": np.array([10])},
        )

    monkeypatch.setattr("app.orchestrator.solve_phaseA_lexicographic", fake_phase_a)
    monkeypatch.setattr("app.orchestrator.audit_phaseA_solution", _audit_stub)

    out = run_two_phase(build_data(), RunConfig("STOCK", "PLAN", 100, time_limit_sec=5))

    assert calls["lex"] == 1
    assert out.run_meta.loc[0, "phase_a_method"] == "lex_once"
    assert out.run_meta.loc[0, "stage1_runtime"] == 1.23


def test_pairs_objective_calls_solver_once(monkeypatch):
    calls = {"solve": 0}

    def fake_solve(*args, **kwargs):
        calls["solve"] += 1
        return SolveOutcome({"FG1": 2, "FG2": 1}, 25.0, "ok", "pairs_once", True, "mip", 2.5)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve)

    out = run_two_phase(build_data(), RunConfig("STOCK", "PAIRS", 100, threads=2))

    assert calls["solve"] == 1
    assert out.run_meta.loc[0, "phase_a_status"] == "ok"
    assert out.run_meta.loc[0, "phase_a_method"] == "mip"
    assert out.run_meta.loc[0, "stage1_runtime"] == 2.5
