from io import BytesIO

import openpyxl

from app.excel_io import build_input_workbook, diagnose_workbook_structure, load_tables_from_excel, write_output_excel
from app.orchestrator import run_optimization
from app.validate import validate_inputs


def test_sample_workbook_round_trip_runs_end_to_end():
    workbook_bytes = build_input_workbook(include_sample_rows=True)

    diagnosis = diagnose_workbook_structure(workbook_bytes)
    assert diagnosis["issues"] == []

    tables = load_tables_from_excel(workbook_bytes)
    validate_inputs(tables)

    fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
        tables,
        run_purchase_planner=True,
        purchase_target_fill_pcts="50,100",
    )
    output_bytes = write_output_excel(
        fg_df,
        rm_df,
        meta_df,
        purchase_summary_df,
        purchase_detail_df,
        purchase_target_sheets=purchase_detail_df.attrs.get("purchase_target_sheets"),
    )

    workbook = openpyxl.load_workbook(BytesIO(output_bytes))
    assert {"FG_Result", "RM_Diagnostic", "Run_Metadata", "Purchase_Summary", "Purchase_Detail", "Purchase_50", "Purchase_100"}.issubset(workbook.sheetnames)
    assert len(purchase_summary_df) == 2
