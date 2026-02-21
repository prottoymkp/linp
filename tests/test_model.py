import numpy as np
import pandas as pd

from app.model import audit_phaseA_solution, solve_optimization, solve_phaseA_lexicographic


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
