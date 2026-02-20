from io import BytesIO

from openpyxl import Workbook
from openpyxl.worksheet.table import Table

from app.excel_io import diagnose_workbook_structure


def _workbook_bytes(with_rows: bool = True) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Input"
    ws.append(["FG Code", "Margin"])
    if with_rows:
        ws.append(["FG1", 12])
    ws.add_table(Table(displayName="tblFG", ref=f"A1:B{2 if with_rows else 1}"))

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_diagnosis_detects_table_and_no_issues_for_valid_shape():
    report = diagnose_workbook_structure(_workbook_bytes(with_rows=True))

    assert report["issues"] == []
    assert report["warnings"] == []
    assert report["tables"] == ["fg_master"]


def test_diagnosis_warns_when_table_has_no_rows():
    report = diagnose_workbook_structure(_workbook_bytes(with_rows=False))

    assert report["issues"] == []
    assert any("no data rows" in w for w in report["warnings"])
