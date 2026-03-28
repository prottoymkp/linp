from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
from openpyxl import Workbook


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import BOM_DATASET, CAP_DATASET, CONTROL_DATASET, FG_DATASET, RM_DATASET
from app.excel_io import _write_df_as_table, load_tables_from_excel, write_output_excel
from app.orchestrator import run_optimization
from app.validate import validate_inputs


SAMPLE_INPUT_FILE = "lp_optimizer_sample_input.xlsx"
SAMPLE_OUTPUT_FILE = "lp_optimizer_sample_output.xlsx"
WORKFLOW_IMAGE_FILE = "lp_optimizer_workflow.svg"
DEMO_TABLES_SUBDIR = "demo_tables"


def _source_shapes(source_workbook: Path | None) -> tuple[int, int, int]:
    if source_workbook is None:
        return 50, 200, 8

    tables = load_tables_from_excel(source_workbook.read_bytes())
    fg_rows = len(tables[FG_DATASET])
    rm_rows = len(tables[RM_DATASET])
    bom_per_fg = max(1, len(tables[BOM_DATASET]) // max(1, fg_rows))
    return fg_rows, rm_rows, bom_per_fg


def _load_demo_tables_from_csv(demo_tables_dir: Path) -> dict[str, pd.DataFrame] | None:
    required_paths = {
        FG_DATASET: demo_tables_dir / f"{FG_DATASET}.csv",
        BOM_DATASET: demo_tables_dir / f"{BOM_DATASET}.csv",
        CAP_DATASET: demo_tables_dir / f"{CAP_DATASET}.csv",
        RM_DATASET: demo_tables_dir / f"{RM_DATASET}.csv",
        CONTROL_DATASET: demo_tables_dir / f"{CONTROL_DATASET}.csv",
    }
    if not all(path.exists() for path in required_paths.values()):
        return None
    return {name: pd.read_csv(path) for name, path in required_paths.items()}


def _sanitize_from_source_workbook(source_workbook: Path) -> dict[str, pd.DataFrame]:
    tables = load_tables_from_excel(source_workbook.read_bytes())

    fg_df = tables[FG_DATASET].copy()
    fg_df["FG Code"] = [f"FG{idx:03d}" for idx in range(1, len(fg_df) + 1)]
    fg_df["Cost Value"] = [
        int(round(float(value) * 0.88 + 21 + (idx % 5) * 3))
        for idx, value in enumerate(fg_df["Cost Value"], start=1)
    ]
    fg_df["Margin"] = [
        int(round(float(value) * 0.93 + 12 + (idx % 4) * 4))
        for idx, value in enumerate(fg_df["Margin"], start=1)
    ]
    fg_df["Dealer Price"] = fg_df["Cost Value"] + fg_df["Margin"]

    fg_code_map = dict(zip(tables[FG_DATASET]["FG Code"].astype(str), fg_df["FG Code"]))

    bom_df = tables[BOM_DATASET].copy()
    bom_df["FG Code"] = bom_df["FG Code"].astype(str).map(fg_code_map)
    bom_df["QtyPerPair"] = [
        round(float(value) * 0.96 + ((idx % 5) * 0.03), 2)
        for idx, value in enumerate(bom_df["QtyPerPair"], start=1)
    ]

    rm_df = tables[RM_DATASET].copy()
    rm_df["RM Code"] = [f"RM{idx:03d}" for idx in range(1, len(rm_df) + 1)]
    rm_code_map = dict(zip(tables[RM_DATASET]["RM Code"].astype(str), rm_df["RM Code"]))
    bom_df["RM Code"] = tables[BOM_DATASET]["RM Code"].astype(str).map(rm_code_map)

    rm_df["Avail_Stock"] = [
        round(float(value) * 0.89 + 35 + idx * 0.4, 2)
        for idx, value in enumerate(rm_df["Avail_Stock"], start=1)
    ]
    rm_df["Avail_StockPO"] = [
        round(float(value) * 0.9 + 45 + idx * 0.5, 2)
        for idx, value in enumerate(rm_df["Avail_StockPO"], start=1)
    ]
    rm_df["RM_Rate"] = [
        int(round(float(value) * 0.87 + 9 + (idx % 3) * 2))
        for idx, value in enumerate(rm_df["RM_Rate"], start=1)
    ]

    cap_df = tables[CAP_DATASET].copy()
    cap_df["FG Code"] = cap_df["FG Code"].astype(str).map(fg_code_map)
    cap_df["Max Plan Qty"] = [
        int(round(float(value) * 0.94 + 5 + (idx % 6) * 2))
        for idx, value in enumerate(cap_df["Max Plan Qty"], start=1)
    ]

    control_df = pd.DataFrame(
        [
            {"Key": "Horizon_Start", "Value": "2026-04-01 00:00:00"},
            {"Key": "Horizon_End", "Value": "2026-04-30 00:00:00"},
            {"Key": "Mode_Avail", "Value": "STOCK"},
            {"Key": "Objective", "Value": "PLAN"},
            {"Key": "PurchaseTargets", "Value": "25,50,75,100"},
        ]
    )

    return {
        FG_DATASET: fg_df,
        BOM_DATASET: bom_df,
        CAP_DATASET: cap_df,
        RM_DATASET: rm_df,
        CONTROL_DATASET: control_df,
    }


def build_sanitized_tables(source_workbook: Path | None = None, demo_tables_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    if source_workbook is not None:
        return _sanitize_from_source_workbook(source_workbook)
    if demo_tables_dir is not None:
        cached_tables = _load_demo_tables_from_csv(demo_tables_dir)
        if cached_tables is not None:
            return cached_tables

    num_fg, num_rm, bom_per_fg = _source_shapes(source_workbook)

    fg_codes = [f"FG{idx:03d}" for idx in range(1, num_fg + 1)]
    rm_codes = [f"RM{idx:03d}" for idx in range(1, num_rm + 1)]

    fg_rows = []
    cap_rows = []
    bom_rows = []

    for fg_idx, fg_code in enumerate(fg_codes, start=1):
        cost_value = 320 + ((fg_idx * 31 + 17) % 190)
        margin = 70 + ((fg_idx * 29 + 11) % 150)
        dealer_price = cost_value + margin
        plan_cap = 80 + ((fg_idx * 19 + 37) % 170)

        fg_rows.append(
            {
                "FG Code": fg_code,
                "Dealer Price": dealer_price,
                "Cost Value": cost_value,
                "Margin": margin,
            }
        )
        cap_rows.append({"FG Code": fg_code, "Max Plan Qty": plan_cap})

        used_rm_indices: set[int] = set()
        for offset in range(bom_per_fg):
            rm_idx = (fg_idx * 7 + offset * 23 + (offset % 3) * 11) % num_rm
            while rm_idx in used_rm_indices:
                rm_idx = (rm_idx + 1) % num_rm
            used_rm_indices.add(rm_idx)
            qty_per_pair = round(0.85 + (((fg_idx + 1) * 13 + (offset + 1) * 17) % 110) / 70.0, 2)
            bom_rows.append(
                {
                    "FG Code": fg_code,
                    "RM Code": rm_codes[rm_idx],
                    "QtyPerPair": qty_per_pair,
                }
            )

    fg_df = pd.DataFrame(fg_rows)
    cap_df = pd.DataFrame(cap_rows)
    bom_df = pd.DataFrame(bom_rows)

    demand = {rm_code: 0.0 for rm_code in rm_codes}
    cap_map = dict(zip(cap_df["FG Code"], cap_df["Max Plan Qty"]))
    for _, row in bom_df.iterrows():
        demand[row["RM Code"]] += float(row["QtyPerPair"]) * float(cap_map[row["FG Code"]])

    rm_rows = []
    for rm_idx, rm_code in enumerate(rm_codes, start=1):
        full_demand = demand[rm_code]
        stock_factor = 0.36 + ((rm_idx * 7 + 5) % 21) / 100.0
        po_factor = stock_factor + 0.18 + ((rm_idx * 5) % 10) / 100.0
        avail_stock = round(max(full_demand * stock_factor, 150.0 + rm_idx * 3.5), 2)
        avail_stock_po = round(max(avail_stock + full_demand * 0.22, full_demand * po_factor), 2)
        rm_rate = 140 + ((rm_idx * 17 + 23) % 135)
        rm_rows.append(
            {
                "RM Code": rm_code,
                "Avail_Stock": avail_stock,
                "Avail_StockPO": avail_stock_po,
                "RM_Rate": rm_rate,
            }
        )

    rm_df = pd.DataFrame(rm_rows)
    control_df = pd.DataFrame(
        [
            {"Key": "Horizon_Start", "Value": "2026-04-01 00:00:00"},
            {"Key": "Horizon_End", "Value": "2026-04-30 00:00:00"},
            {"Key": "Mode_Avail", "Value": "STOCK"},
            {"Key": "Objective", "Value": "PLAN"},
            {"Key": "PurchaseTargets", "Value": "25,50,75,100"},
        ]
    )

    return {
        FG_DATASET: fg_df,
        BOM_DATASET: bom_df,
        CAP_DATASET: cap_df,
        RM_DATASET: rm_df,
        CONTROL_DATASET: control_df,
    }


def build_sample_input_workbook(tables: dict[str, pd.DataFrame]) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)

    sheet_specs = [
        ("tblFG", "tblFG", tables[FG_DATASET]),
        ("tblBOM", "tblBOM", tables[BOM_DATASET]),
        ("tblFGPlanCap", "tblFGPlanCap", tables[CAP_DATASET]),
        ("tblRMAvail", "tblRMAvail", tables[RM_DATASET]),
        ("Solver_Control", "tblControl_2", tables[CONTROL_DATASET]),
    ]

    for sheet_name, table_name, df in sheet_specs:
        ws = wb.create_sheet(sheet_name)
        _write_df_as_table(ws, df, table_name)

    payload = BytesIO()
    wb.save(payload)
    payload.seek(0)
    return payload.getvalue()


def write_demo_table_sources(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, frame in tables.items():
        (output_dir / f"{table_name}.csv").write_text(frame.to_csv(index=False), encoding="utf-8", newline="")


def build_workflow_preview(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 860" role="img" aria-label="LP Optimizer workflow preview">
  <rect width="1500" height="860" fill="#f7f4ee"/>
  <rect x="40" y="30" width="1420" height="800" rx="26" fill="#fffdf8" stroke="#d9cdbb" stroke-width="3"/>
  <text x="70" y="82" fill="#183153" font-size="28" font-family="Segoe UI, Arial, sans-serif" font-weight="700">LP Optimizer Workflow Preview</text>
  <text x="70" y="116" fill="#5d6f82" font-size="18" font-family="Segoe UI, Arial, sans-serif">Input workbook -&gt; Run optimization -&gt; Output workbook</text>
  <line x1="470" y1="455" x2="520" y2="455" stroke="#ba8c4f" stroke-width="6"/>
  <polygon points="520,455 502,443 502,467" fill="#ba8c4f"/>
  <line x1="945" y1="455" x2="995" y2="455" stroke="#ba8c4f" stroke-width="6"/>
  <polygon points="995,455 977,443 977,467" fill="#ba8c4f"/>
  <rect x="80" y="160" width="390" height="600" rx="22" fill="#eef4ff" stroke="#d9cdbb" stroke-width="2"/>
  <text x="104" y="208" fill="#183153" font-size="22" font-family="Segoe UI, Arial, sans-serif" font-weight="700">1. Start with a sample input</text>
  <text x="104" y="256" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Workbook: generated sample input</text>
  <text x="104" y="302" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">FG table rows: {len(tables[FG_DATASET])}</text>
  <text x="104" y="348" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">BOM table rows: {len(tables[BOM_DATASET])}</text>
  <text x="104" y="394" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">RM rows: {len(tables[RM_DATASET])}</text>
  <text x="104" y="440" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Named tables: tblFG, tblBOM,</text>
  <text x="104" y="486" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">tblFGPlanCap, tblRMAvail,</text>
  <text x="104" y="532" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">and tblControl_2</text>
  <rect x="555" y="160" width="390" height="600" rx="22" fill="#eff9f0" stroke="#d9cdbb" stroke-width="2"/>
  <text x="579" y="208" fill="#183153" font-size="22" font-family="Segoe UI, Arial, sans-serif" font-weight="700">2. Run the Streamlit app</text>
  <text x="579" y="256" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Upload the workbook</text>
  <text x="579" y="302" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Validate table structure</text>
  <text x="579" y="348" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Choose purchase targets</text>
  <text x="579" y="394" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Adjust optional solver settings</text>
  <text x="579" y="440" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Run optimization and review progress</text>
  <rect x="1030" y="160" width="390" height="600" rx="22" fill="#fff2e8" stroke="#d9cdbb" stroke-width="2"/>
  <text x="1054" y="208" fill="#183153" font-size="22" font-family="Segoe UI, Arial, sans-serif" font-weight="700">3. Review the output workbook</text>
  <text x="1054" y="256" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">FG_Result: production recommendation</text>
  <text x="1054" y="302" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">RM_Diagnostic: bottleneck materials</text>
  <text x="1054" y="348" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Purchase_Summary: target vs buy cost</text>
  <text x="1054" y="394" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Run_Metadata: solver trace</text>
  <text x="1054" y="440" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">Purchase_Detail and Purchase_&lt;target&gt;</text>
</svg>"""
    (output_dir / WORKFLOW_IMAGE_FILE).write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bundled demo assets for the Streamlit app.")
    parser.add_argument("--source-workbook", type=Path, help="Optional source workbook used only to mirror public sample scale.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "assets", help="Directory for generated demo assets.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = build_sanitized_tables(args.source_workbook, output_dir / DEMO_TABLES_SUBDIR)
    validate_inputs(tables)
    write_demo_table_sources(output_dir / DEMO_TABLES_SUBDIR, tables)

    sample_input = build_sample_input_workbook(tables)
    (output_dir / SAMPLE_INPUT_FILE).write_bytes(sample_input)

    fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
        tables,
        run_purchase_planner=True,
    )
    sample_output = write_output_excel(
        fg_df,
        rm_df,
        meta_df,
        purchase_summary_df,
        purchase_detail_df,
        purchase_target_sheets=purchase_detail_df.attrs.get("purchase_target_sheets"),
    )
    (output_dir / SAMPLE_OUTPUT_FILE).write_bytes(sample_output)

    build_workflow_preview(output_dir, tables)

    print(f"Wrote {(output_dir / DEMO_TABLES_SUBDIR)}")
    print(f"Wrote {(output_dir / SAMPLE_INPUT_FILE)}")
    print(f"Wrote {(output_dir / SAMPLE_OUTPUT_FILE)}")
    print(f"Wrote {(output_dir / WORKFLOW_IMAGE_FILE)}")


if __name__ == "__main__":
    main()
