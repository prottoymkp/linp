"""Streamlit UI for upload, validate, run and download."""

from __future__ import annotations

import tempfile

import streamlit as st

from app.excel_io import load_input_tables, write_output_workbook
from app.orchestrator import run_two_phase
from app.validate import ValidationError, validate_inputs


st.set_page_config(page_title="FG Optimizer", layout="wide")
st.title("FG Optimization Runner")

uploaded = st.file_uploader("Upload input workbook (.xlsx)", type=["xlsx"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        f.write(uploaded.getbuffer())
        input_path = f.name

    try:
        data = load_input_tables(input_path)
        config = validate_inputs(data)
        st.success("Validation passed.")
        if st.button("Run Optimization"):
            outputs = run_two_phase(data, config)
            total_pairs = int(outputs.fg_result["Opt Qty Total"].sum())
            total_margin = float(outputs.fg_result["Total Margin"].sum())
            binding_rm = int((outputs.rm_diagnostic["remaining_availability"] <= 1e-9).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Pairs", f"{total_pairs:,}")
            c2.metric("Total Margin", f"{total_margin:,.2f}")
            c3.metric("Binding RM Count", f"{binding_rm}")

            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as out:
                write_output_workbook(out.name, outputs.fg_result, outputs.rm_diagnostic, outputs.run_meta)
                out.seek(0)
                payload = out.read()

            st.download_button(
                "Download output workbook",
                data=payload,
                file_name="optimization_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.dataframe(outputs.fg_result)
            st.dataframe(outputs.rm_diagnostic)
    except ValidationError as exc:
        st.error("Validation failed.")
        for err in exc.errors:
            st.write(f"- {err}")
    except Exception as exc:  # pragma: no cover
        st.exception(exc)
