import numpy as np
import pandas as pd
import pytest

from app.config import BOM_DATASET, CAP_DATASET, FG_DATASET, RM_DATASET
from app.orchestrator import run_two_phase
from app.types import RunConfig, SolveOutcome


def _tables():
    return {
        FG_DATASET: pd.DataFrame({"FG Code": ["FG1", "FG2"], "Unit Margin": [10, 5]}),
        BOM_DATASET: pd.DataFrame({"FG Code": ["FG1", "FG2"], "RM Code": ["RM1", "RM1"], "QtyPerPair": [1, 1]}),
        CAP_DATASET: pd.DataFrame({"FG Code": ["FG1", "FG2"], "Plan Cap": [2, 2]}),
        RM_DATASET: pd.DataFrame({"RM Code": ["RM1"], "Avail_Stock": [4], "Avail_StockPO": [4]}),
    }


def test_phase_a_plan_routes_to_lex(monkeypatch):
    called = {"lex": 0, "single": 0}

    def fake_lex(*args, **kwargs):
        called["lex"] += 1
        return (
            SolveOutcome({"FG1": 2, "FG2": 2}, 30.0, "Optimal", "highs", False, "lex_mip", 0.01),
            {"stage1_status": "Optimal", "stage2_status": "Optimal", "P_star": 4, "method": "lex_mip"},
            {"x_stage1": np.array([2, 2]), "x_stage2": np.array([2, 2]), "coeff": np.array([[1.0, 1.0]]), "caps": np.array([2.0, 2.0]), "rm_upper": np.array([4.0])},
        )

    def fake_single(*args, **kwargs):
        called["single"] += 1
        return SolveOutcome({"FG1": 0, "FG2": 0}, 0.0, "Optimal", "highs", False, "mip", 0.01)

    monkeypatch.setattr("app.orchestrator.solve_phaseA_lexicographic", fake_lex)
    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_single)
    out = run_two_phase(_tables(), RunConfig("STOCK", "PLAN", 100))

    assert called["lex"] == 1
    assert out.run_meta.loc[0, "objective"] == "PLAN"


def test_phase_a_margin_routes_to_single_objective(monkeypatch):
    called = {"lex": 0, "single": 0}

    def fake_lex(*args, **kwargs):
        called["lex"] += 1
        raise AssertionError("should not call lex")

    def fake_single(*args, **kwargs):
        called["single"] += 1
        return SolveOutcome({"FG1": 2, "FG2": 2}, 30.0, "Optimal", "highs", False, "mip", 0.01)

    monkeypatch.setattr("app.orchestrator.solve_phaseA_lexicographic", fake_lex)
    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_single)
    out = run_two_phase(_tables(), RunConfig("STOCK", "MARGIN", 100))

    assert called["single"] >= 1
    assert out.run_meta.loc[0, "objective"] == "MARGIN"
    assert out.run_meta.loc[0, "stage2_status"] == "n/a"


def test_invalid_objective_raises():
    with pytest.raises(ValueError):
        run_two_phase(_tables(), RunConfig("STOCK", "NOPE", 100))
