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


def test_phase_b_zero_when_any_cap_unmet(monkeypatch):
    calls = {"n": 0}

    def fake_solve(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SolveOutcome({"FG1": 2, "FG2": 1}, 25.0, "ok", "mock", False, "mip", 0.01)
        return SolveOutcome({"FG1": 1, "FG2": 1}, 15.0, "ok", "mock", False, "mip", 0.01)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve)
    out = run_two_phase(build_data(), RunConfig("STOCK", "MARGIN", 100))

    assert out.fg_result["Opt Qty Phase B"].sum() == 0
    assert out.run_meta.loc[0, "phase_b_executed"] == False
    assert calls["n"] == 1


def test_phase_b_enabled_when_all_caps_met(monkeypatch):
    calls = {"n": 0}

    def fake_solve(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return SolveOutcome({"FG1": 2, "FG2": 2}, 30.0, "ok", "mock", False, "mip", 0.01)
        return SolveOutcome({"FG1": 1, "FG2": 0}, 10.0, "ok", "mock", False, "mip", 0.01)

    monkeypatch.setattr("app.orchestrator.solve_optimization", fake_solve)
    out = run_two_phase(build_data(), RunConfig("STOCK", "MARGIN", 100))

    assert out.fg_result["Opt Qty Phase B"].sum() == 1
    assert out.run_meta.loc[0, "phase_b_executed"] == True
    assert calls["n"] == 2
