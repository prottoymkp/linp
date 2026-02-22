import numpy as np
import pandas as pd

from app.model import (
    _fallback_stage2_reallocate,
    _greedy_fill,
    audit_phaseA_solution,
    solve_optimization,
    solve_phaseA_lexicographic,
    solve_purchase_planner_milp,
)


def test_rm_feasibility_and_integer_outputs():
    fg_df = pd.DataFrame({"FG Code": ["FG1", "FG2"], "Unit Margin": [10, 8]})
    bom_df = pd.DataFrame(
        {
            "FG Code": ["FG1", "FG2"],
            "RM Code": ["RM1", "RM1"],
            "QtyPerPair": [2, 3],
        }
    )
    cap_df = pd.DataFrame({"FG Code": ["FG1", "FG2"], "Plan Cap": [10, 10]})
    rm_df = pd.DataFrame({"RM Code": ["RM1"], "Avail_Stock": [15], "Avail_StockPO": [15]})

    out = solve_optimization(fg_df, bom_df, cap_df, rm_df, mode_avail="STOCK", objective="MARGIN")
    q1, q2 = out.quantities["FG1"], out.quantities["FG2"]

    assert isinstance(q1, int) and isinstance(q2, int)
    assert 2 * q1 + 3 * q2 <= 15 + 1e-9
    assert q1 >= 0 and q2 >= 0
    assert np.isfinite(out.objective_value)


def test_phase_a_lexicographic_keeps_pairs_then_maximizes_margin():
    fg_df = pd.DataFrame({"FG Code": ["A", "B"], "Unit Margin": [5, 10]})
    bom_df = pd.DataFrame({"FG Code": ["A", "B"], "RM Code": ["R1", "R1"], "QtyPerPair": [1, 2]})
    cap_df = pd.DataFrame({"FG Code": ["A", "B"], "Plan Cap": [2, 2]})
    rm_df = pd.DataFrame({"RM Code": ["R1"], "Avail_Stock": [2], "Avail_StockPO": [2]})

    out, meta, audit = solve_phaseA_lexicographic(fg_df, bom_df, cap_df, rm_df, mode_avail="STOCK")

    # Stage-1 max pairs is 2, stage-2 should select A=2 over B=1 due to pair lock and higher total margin.
    assert out.quantities == {"A": 2, "B": 0}
    assert meta["P_star"] == 2
    assert meta["stage1_status"] in {"Optimal", "Feasible"}
    assert meta["stage2_status"] in {"Optimal", "Feasible"}
    audit_phaseA_solution(audit, stage2_success=True)


def test_solve_optimization_rejects_plan_objective():
    fg_df = pd.DataFrame({"FG Code": ["FG1"], "Unit Margin": [10]})
    bom_df = pd.DataFrame({"FG Code": ["FG1"], "RM Code": ["RM1"], "QtyPerPair": [1]})
    cap_df = pd.DataFrame({"FG Code": ["FG1"], "Plan Cap": [5]})
    rm_df = pd.DataFrame({"RM Code": ["RM1"], "Avail_Stock": [10], "Avail_StockPO": [10]})

    import pytest

    with pytest.raises(ValueError):
        solve_optimization(fg_df, bom_df, cap_df, rm_df, mode_avail="STOCK", objective="PLAN")


def test_purchase_planner_pairs_shortage_objective_and_integrality():
    fg_df = pd.DataFrame({"FG Code": ["A"], "Unit Margin": [5]})
    bom_df = pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [1]})
    cap_df = pd.DataFrame({"FG Code": ["A"], "Plan Cap": [5]})
    rm_df = pd.DataFrame(
        {
            "RM Code": ["R1"],
            "Avail_Stock": [2],
            "Avail_StockPO": [2],
            "RM_Rate": [3],
        }
    )

    out = solve_purchase_planner_milp(fg_df, bom_df, cap_df, rm_df, mode_avail="STOCK", scenario="PAIRS", target_value=4)

    assert out["status"] in {"Optimal", "Feasible"}
    assert out["x"] == {"A": 4}
    assert out["y"]["R1"] == 2.0
    assert out["objective_value"] == 6.0


def test_purchase_planner_margin_at_pair_fill_respects_mode_avail():
    fg_df = pd.DataFrame({"FG Code": ["A"], "Unit Margin": [10]})
    bom_df = pd.DataFrame({"FG Code": ["A"], "RM Code": ["R1"], "QtyPerPair": [1]})
    cap_df = pd.DataFrame({"FG Code": ["A"], "Plan Cap": [10]})
    rm_df = pd.DataFrame(
        {
            "RM Code": ["R1"],
            "Avail_Stock": [0],
            "Avail_StockPO": [5],
            "RM_Rate": [2],
        }
    )

    out = solve_purchase_planner_milp(
        fg_df,
        bom_df,
        cap_df,
        rm_df,
        mode_avail="STOCK_PO",
        scenario="MARGIN_AT_PAIR_FILL",
        target_value=50,
    )

    assert out["status"] in {"Optimal", "Feasible"}
    assert out["x"]["A"] == 5
    assert out["y"]["R1"] == 0.0
    assert out["objective_value"] == 0.0


def test_greedy_fill_respects_iteration_cutoff():
    base = np.array([0], dtype=int)
    caps = np.array([100], dtype=float)
    coeff = np.array([[0.0]], dtype=float)
    rm_upper = np.array([100.0], dtype=float)
    scores = np.array([1.0], dtype=float)

    x, meta = _greedy_fill(base, 100, caps, coeff, rm_upper, scores, max_iterations=2, max_wall_time_sec=10.0)

    assert int(x.sum()) == 2
    assert meta["heuristic_cutoff_hit"] is True
    assert meta["cutoff_reason"] == "max_iterations"


def test_fallback_reallocate_reports_cutoff_metadata():
    x_stage1 = np.array([5, 5], dtype=int)
    margins = np.array([10.0, 1.0], dtype=float)
    caps = np.array([10.0, 10.0], dtype=float)
    coeff = np.array([[1.0, 1.0]], dtype=float)
    rm_upper = np.array([10.0], dtype=float)

    _, meta = _fallback_stage2_reallocate(
        x_stage1,
        margins,
        caps,
        coeff,
        rm_upper,
        max_passes=1,
        max_swaps=100,
        max_wall_time_sec=10.0,
    )

    assert meta["passes"] == 1
    assert meta["heuristic_cutoff_hit"] is True
    assert meta["cutoff_reason"] == "max_passes"


def test_solve_optimization_fallback_large_feasible_caps_reaches_target_or_reports_cutoff(monkeypatch):
    import app.model as model_mod

    n_fg = 600
    fg_codes = [f"FG{i}" for i in range(n_fg)]
    fg_df = pd.DataFrame({"FG Code": fg_codes, "Unit Margin": np.ones(n_fg)})
    bom_df = pd.DataFrame({"FG Code": fg_codes, "RM Code": ["RM1"] * n_fg, "QtyPerPair": np.ones(n_fg)})
    cap_df = pd.DataFrame({"FG Code": fg_codes, "Plan Cap": np.ones(n_fg)})
    rm_df = pd.DataFrame({"RM Code": ["RM1"], "Avail_Stock": [float(n_fg)], "Avail_StockPO": [float(n_fg)]})

    call_state = {"count": 0}

    def fake_solve_single_objective(model_inputs, objective, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return "Unknown", np.zeros(len(objective), dtype=float), 0.0, 0.0
        return "Optimal", np.full(len(objective), 0.999, dtype=float), float(0.999 * len(objective)), 0.0

    monkeypatch.setattr(model_mod, "_solve_single_objective", fake_solve_single_objective)

    out = model_mod.solve_optimization(
        fg_df,
        bom_df,
        cap_df,
        rm_df,
        mode_avail="STOCK",
        objective="PAIRS",
        enforce_caps=True,
    )

    total = sum(out.quantities.values())
    if total < n_fg:
        assert out.used_fallback is True
        assert out.heuristic_cutoff_hit is True
        assert "cutoff" in out.status
    else:
        assert total == n_fg
        assert "target_reached" in out.status
