"""Excel read/write helpers for named table ingestion and result export."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.utils.cell import range_boundaries
from openpyxl.worksheet.table import Table, TableStyleInfo

from app.config import TABLE_ALIASES


class ExcelIOError(Exception):
    """Raised when workbook ingestion/writing fails."""


def _table_to_dataframe(ws, table) -> pd.DataFrame:
    min_col, min_row, max_col, max_row = range_boundaries(table.ref)
    rows = ws.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col, values_only=True)
    rows = list(rows)
    if not rows:
        return pd.DataFrame()
    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    data = rows[1:]
    return pd.DataFrame(data, columns=headers)


def load_input_tables(path: str | Path) -> Dict[str, pd.DataFrame]:
    """Load workbook tables into logical datasets with alias normalization."""
    wb = load_workbook(filename=path, data_only=True)
    datasets: Dict[str, pd.DataFrame] = {}
    for ws in wb.worksheets:
        for table in ws.tables.values():
            logical_name = TABLE_ALIASES.get(table.name)
            if logical_name:
                datasets[logical_name] = _table_to_dataframe(ws, table)
    return datasets


def _write_table_sheet(wb: Workbook, sheet_name: str, table_name: str, df: pd.DataFrame) -> None:
    ws = wb.create_sheet(title=sheet_name)
    ws.append(list(df.columns))
    for row in df.itertuples(index=False):
        ws.append(list(row))

    end_col_letter = ws.cell(row=1, column=max(1, len(df.columns))).column_letter
    end_row = max(2, len(df) + 1)
    ref = f"A1:{end_col_letter}{end_row}"

    table = Table(displayName=table_name, ref=ref)
    style = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True)
    table.tableStyleInfo = style
    ws.add_table(table)


def write_output_workbook(path: str | Path, fg_result: pd.DataFrame, rm_diagnostic: pd.DataFrame, run_meta: pd.DataFrame) -> None:
    """Write output workbook with stable sheet/table names and column order."""
    wb = Workbook()
    wb.remove(wb.active)

    _write_table_sheet(wb, "FGResult", "tblFGResult", fg_result)
    _write_table_sheet(wb, "RMDiagnostic", "tblRMDiagnostic", rm_diagnostic)
    _write_table_sheet(wb, "RunMeta", "tblRunMeta", run_meta)

    wb.save(path)
