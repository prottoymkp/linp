import numpy as np
import pandas as pd

from app.model import solve_optimization


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
