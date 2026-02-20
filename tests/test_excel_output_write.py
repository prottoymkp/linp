from io import BytesIO

import openpyxl
import pandas as pd

from app.excel_io import write_output_excel


def test_write_output_excel_writes_single_header_and_data_block_per_sheet():
    fg_df = pd.DataFrame(
        [
            {"FG Code": "FG1", "Units": 10},
            {"FG Code": "FG2", "Units": 20},
        ]
    )
    rm_df = pd.DataFrame([
        {"RM Code": "RM1", "Variance": 1.5},
    ])
    meta_df = pd.DataFrame([
        {"Metric": "run_id", "Value": "abc123"},
    ])

    output_bytes = write_output_excel(fg_df=fg_df, rm_df=rm_df, meta_df=meta_df)
    wb = openpyxl.load_workbook(BytesIO(output_bytes))

    expected = [
        ("FG_Result", fg_df, "tblFGResult"),
        ("RM_Diagnostic", rm_df, "tblRMDiagnostic"),
        ("Run_Metadata", meta_df, "tblRunMeta"),
    ]

    for sheet_name, df, table_name in expected:
        ws = wb[sheet_name]

        # Regression check: this used to be duplicated (2 * len(df) + 2 rows)
        # because the dataframe was written once by pandas and appended again.
        assert ws.max_row == len(df) + 1

        header_row = [cell.value for cell in ws[1]]
        assert header_row == df.columns.tolist()

        assert len(ws.tables) == 1
        table = next(iter(ws.tables.values()))
        assert table.displayName == table_name
