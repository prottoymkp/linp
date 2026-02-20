from __future__ import annotations

from io import BytesIO
from typing import Dict

import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

from .config import TABLE_ALIASES


def load_tables_from_excel(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    wb = load_workbook(BytesIO(file_bytes), data_only=True)
    tables: Dict[str, pd.DataFrame] = {}

    for ws in wb.worksheets:
        for table in ws.tables.values():
            ref = ws[table.ref]
            data = [[c.value for c in row] for row in ref]
            if not data:
                continue
            headers = [str(h) for h in data[0]]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
            tname = TABLE_ALIASES.get(table.name, table.name)
            tables[tname] = df

    return tables


def _write_df_as_table(ws, df: pd.DataFrame, table_name: str):
    ws.append(df.columns.tolist())
    for row in df.itertuples(index=False):
        ws.append(list(row))
    end_col = chr(ord("A") + len(df.columns) - 1)
    end_row = len(df) + 1
    tab = Table(displayName=table_name, ref=f"A1:{end_col}{end_row}")
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    tab.tableStyleInfo = style
    ws.add_table(tab)


def write_output_excel(fg_df: pd.DataFrame, rm_df: pd.DataFrame, meta_df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        fg_df.to_excel(writer, sheet_name="FG_Result", index=False)
        rm_df.to_excel(writer, sheet_name="RM_Diagnostic", index=False)
        meta_df.to_excel(writer, sheet_name="Run_Metadata", index=False)

    out.seek(0)
    wb = load_workbook(out)
    ws_fg = wb["FG_Result"]
    ws_rm = wb["RM_Diagnostic"]
    ws_meta = wb["Run_Metadata"]

    for ws in [ws_fg, ws_rm, ws_meta]:
        if ws.tables:
            for name in list(ws.tables.keys()):
                del ws.tables[name]

    _write_df_as_table(ws_fg, fg_df, "tblFGResult")
    _write_df_as_table(ws_rm, rm_df, "tblRMDiagnostic")
    _write_df_as_table(ws_meta, meta_df, "tblRunMeta")

    final = BytesIO()
    wb.save(final)
    final.seek(0)
    return final.getvalue()
