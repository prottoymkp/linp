import pandas as pd

from app.orchestrator import generate_purchase_planning_scenarios


def test_generate_purchase_planning_scenarios_targets_and_fields():
    fg = pd.DataFrame({"FG Code": ["A", "B"], "Margin": [3, 2]})
    cap = pd.DataFrame({"FG Code": ["A", "B"], "Max Plan Qty": [4, 6]})

    out = generate_purchase_planning_scenarios(fg, cap, mode_avail="STOCK")

    assert list(out.columns) == ["target_metric", "fill_pct", "target_value", "mode_avail", "status"]
    assert len(out) == 8

    pairs = out[out["target_metric"] == "PAIRS"].reset_index(drop=True)
    assert pairs["target_value"].tolist() == [3, 5, 8, 10]
    assert set(pairs["mode_avail"]) == {"STOCK"}
    assert set(pairs["status"]) == {"runnable"}

    margin = out[out["target_metric"] == "MARGIN_AT_PAIR_FILL"].reset_index(drop=True)
    assert margin["target_value"].tolist() == [6.0, 12.0, 18.0, 24.0]
    assert set(margin["status"]) == {"runnable"}


def test_generate_purchase_planning_scenarios_nonpositive_plan_margin_status():
    fg = pd.DataFrame({"FG Code": ["A", "B"], "Margin": [-1, 0]})
    cap = pd.DataFrame({"FG Code": ["A", "B"], "Max Plan Qty": [4, 6]})

    out = generate_purchase_planning_scenarios(fg, cap, mode_avail="STOCK")

    margin = out[out["target_metric"] == "MARGIN_AT_PAIR_FILL"]
    assert set(margin["status"]) == {"not_run_plan_margin_nonpositive"}
