from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from app.config import BOM_DATASET, CAP_DATASET, CONTROL_DATASET, FG_DATASET, RM_DATASET
from app.excel_io import _write_df_as_table, write_output_excel
from app.orchestrator import run_optimization
from app.validate import validate_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
DEMO_TABLES_DIR = ASSETS_DIR / "demo_tables"
SAMPLE_INPUT_PATH = ASSETS_DIR / "lp_optimizer_sample_input.xlsx"
SAMPLE_OUTPUT_PATH = ASSETS_DIR / "lp_optimizer_sample_output.xlsx"
WORKFLOW_PREVIEW_PATH = ASSETS_DIR / "lp_optimizer_workflow.svg"

_DEMO_TABLE_NAMES = (
    FG_DATASET,
    BOM_DATASET,
    CAP_DATASET,
    RM_DATASET,
    CONTROL_DATASET,
)


@lru_cache(maxsize=None)
def read_asset_bytes(path: Path) -> bytes:
    return path.read_bytes()


@lru_cache(maxsize=None)
def read_asset_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _copy_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: df.copy() for name, df in tables.items()}


@lru_cache(maxsize=None)
def _demo_tables_cached() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for table_name in _DEMO_TABLE_NAMES:
        path = DEMO_TABLES_DIR / f"{table_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing demo table source: {path}")
        tables[table_name] = pd.read_csv(path)
    return tables


def load_demo_tables() -> dict[str, pd.DataFrame]:
    return _copy_tables(_demo_tables_cached())


def _build_sample_input_workbook(tables: dict[str, pd.DataFrame]) -> bytes:
    workbook = Workbook()
    workbook.remove(workbook.active)

    sheet_specs = [
        ("tblFG", "tblFG", tables[FG_DATASET]),
        ("tblBOM", "tblBOM", tables[BOM_DATASET]),
        ("tblFGPlanCap", "tblFGPlanCap", tables[CAP_DATASET]),
        ("tblRMAvail", "tblRMAvail", tables[RM_DATASET]),
        ("Solver_Control", "tblControl_2", tables[CONTROL_DATASET]),
    ]

    for sheet_name, table_name, frame in sheet_specs:
        worksheet = workbook.create_sheet(sheet_name)
        _write_df_as_table(worksheet, frame, table_name)

    payload = BytesIO()
    workbook.save(payload)
    payload.seek(0)
    return payload.getvalue()


@lru_cache(maxsize=None)
def sample_input_bytes() -> bytes:
    if SAMPLE_INPUT_PATH.exists():
        return read_asset_bytes(SAMPLE_INPUT_PATH)

    tables = load_demo_tables()
    validate_inputs(tables)
    return _build_sample_input_workbook(tables)


@lru_cache(maxsize=None)
def sample_output_bytes() -> bytes:
    if SAMPLE_OUTPUT_PATH.exists():
        return read_asset_bytes(SAMPLE_OUTPUT_PATH)

    tables = load_demo_tables()
    validate_inputs(tables)
    fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
        tables,
        run_purchase_planner=True,
        purchase_target_fill_pcts="25,50,75,100",
    )
    return write_output_excel(
        fg_df,
        rm_df,
        meta_df,
        purchase_summary_df,
        purchase_detail_df,
        purchase_target_sheets=purchase_detail_df.attrs.get("purchase_target_sheets"),
    )


def _build_workflow_preview_svg() -> str:
    tables = load_demo_tables()
    fg_rows = len(tables[FG_DATASET])
    bom_rows = len(tables[BOM_DATASET])
    rm_rows = len(tables[RM_DATASET])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1500 860" role="img" aria-label="LP Optimizer workflow preview" style="width:100%;height:auto;display:block">
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
  <text x="104" y="302" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">FG table rows: {fg_rows}</text>
  <text x="104" y="348" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">BOM table rows: {bom_rows}</text>
  <text x="104" y="394" fill="#334e68" font-size="18" font-family="Segoe UI, Arial, sans-serif">RM rows: {rm_rows}</text>
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


@lru_cache(maxsize=None)
def workflow_preview_svg() -> str:
    if WORKFLOW_PREVIEW_PATH.exists():
        return read_asset_text(WORKFLOW_PREVIEW_PATH)
    return _build_workflow_preview_svg()
