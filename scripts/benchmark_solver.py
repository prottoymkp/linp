from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.excel_io import load_tables_from_excel
from app.orchestrator import run_optimization
from app.validate import validate_inputs


def build_tables(num_fg: int, num_rm: int, bom_per_fg: int) -> dict[str, pd.DataFrame]:
    fg_codes = [f"FG_{idx:04d}" for idx in range(num_fg)]
    rm_codes = [f"RM_{idx:04d}" for idx in range(num_rm)]

    fg_master = pd.DataFrame(
        {
            "FG Code": fg_codes,
            "Margin": np.linspace(5.0, 15.0, num_fg),
        }
    )
    fg_caps = pd.DataFrame(
        {
            "FG Code": fg_codes,
            "Max Plan Qty": np.full(num_fg, 100, dtype=int),
        }
    )

    bom_rows: list[dict[str, object]] = []
    for fg_index, fg_code in enumerate(fg_codes):
        for offset in range(bom_per_fg):
            rm_code = rm_codes[(fg_index + offset) % num_rm]
            bom_rows.append({"FG Code": fg_code, "RM Code": rm_code, "QtyPerPair": 1.0 + (offset % 3)})
    bom_master = pd.DataFrame(bom_rows)

    rm_avail = pd.DataFrame(
        {
            "RM Code": rm_codes,
            "Avail_Stock": np.full(num_rm, num_fg * 12.0 / max(1, num_rm)),
            "Avail_StockPO": np.full(num_rm, num_fg * 15.0 / max(1, num_rm)),
            "RM_Rate": np.linspace(1.5, 4.5, num_rm),
        }
    )

    control = pd.DataFrame(
        {
            "Key": ["Mode_Avail", "Objective", "PurchaseTargets"],
            "Value": ["STOCK", "PLAN", "50,100"],
        }
    )

    return {
        "fg_master": fg_master,
        "bom_master": bom_master,
        "tblFGPlanCap": fg_caps,
        "tblRMAvail": rm_avail,
        "tblControl_2": control,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the optimization flow on synthetic data.")
    parser.add_argument("--workbook", type=Path, help="Optional .xlsx workbook to benchmark instead of synthetic data.")
    parser.add_argument("--fg", type=int, default=250, help="Number of FG rows.")
    parser.add_argument("--rm", type=int, default=40, help="Number of RM rows.")
    parser.add_argument("--bom-per-fg", type=int, default=3, help="Average BOM rows per FG.")
    parser.add_argument("--purchase-planner", action="store_true", help="Include purchase planning scenarios.")
    args = parser.parse_args()

    if args.workbook is not None:
        raw = args.workbook.read_bytes()
        tables = load_tables_from_excel(raw)
        validate_inputs(tables)
        source_label = str(args.workbook)
    else:
        tables = build_tables(num_fg=args.fg, num_rm=args.rm, bom_per_fg=args.bom_per_fg)
        source_label = f"synthetic fg={args.fg} rm={args.rm} bom_per_fg={args.bom_per_fg}"

    started = time.perf_counter()
    fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
        tables,
        run_purchase_planner=args.purchase_planner,
    )
    elapsed = time.perf_counter() - started

    meta_map = dict(zip(meta_df["Key"], meta_df["Value"]))
    print(f"source={source_label}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"fg_rows={len(fg_df)}")
    print(f"rm_rows={len(rm_df)}")
    print(f"purchase_summary_rows={len(purchase_summary_df)}")
    print(f"purchase_detail_rows={len(purchase_detail_df)}")
    print(f"phase_a_status={meta_map.get('phase_a_status')}")
    print(f"phase_b_status={meta_map.get('phase_b_status')}")
    print(f"threads={meta_map.get('threads')}")


if __name__ == "__main__":
    main()
