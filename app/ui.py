from pathlib import Path
import sys

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import REQUIRED_TABLES
from app.excel_io import diagnose_workbook_structure, load_tables_from_excel, write_output_excel
from app.orchestrator import run_optimization
from app.validate import ValidationError, validate_inputs


st.set_page_config(page_title="LP Optimizer Service", page_icon="📈", layout="centered")
st.title("LP Optimizer Service for RM-Constrained FG Planning")

with st.expander("Input file structure requirements", expanded=True):
    st.markdown("Upload an **.xlsx workbook** that contains named Excel Tables with the following structures.")
    st.caption("Table aliases accepted: tblFG → fg_master, tblBOM → bom_master")

    for table_name, required_columns in REQUIRED_TABLES.items():
        cols = ", ".join(required_columns) if required_columns else "No mandatory columns, but must include control key/value pairs."
        st.write(f"- **{table_name}**: {cols}")

    st.info("Optional accepted columns: fg_master accepts Margin or Unit Margin; tblFGPlanCap accepts Max Plan Qty or Plan Cap.")

upload = st.file_uploader("Upload input Excel (.xlsx)", type=["xlsx"])

if upload is not None:
    raw = upload.read()
    try:
        st.subheader("File quality diagnosis")
        diagnosis = diagnose_workbook_structure(raw)

        if diagnosis["issues"]:
            for issue in diagnosis["issues"]:
                st.error(issue)
        if diagnosis["warnings"]:
            for warning in diagnosis["warnings"]:
                st.warning(warning)
        if diagnosis["tables"]:
            st.write({"Detected tables": diagnosis["tables"]})

        if diagnosis["issues"]:
            st.stop()

        tables = load_tables_from_excel(raw)
        validate_inputs(tables)
        st.success("Validation passed.")

        run_purchase_planner = st.checkbox("Run purchase planner (25/50/75/100)", value=True)

        if st.button("Run Optimization", type="primary"):
            fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
                tables,
                run_purchase_planner=run_purchase_planner,
            )
            out_bytes = write_output_excel(fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df, purchase_target_sheets=purchase_detail_df.attrs.get("purchase_target_sheets"))

            st.subheader("Summary")
            st.write({
                "Total Pairs": int(fg_df["Opt Qty Total"].sum()),
                "Total Margin": float(fg_df["Total Margin"].sum()),
                "RM rows": int(len(rm_df)),
            })

            if run_purchase_planner and not purchase_summary_df.empty:
                st.subheader("Purchase summary preview")
                st.dataframe(purchase_summary_df, use_container_width=True)

            st.download_button(
                "Download Optimized Output",
                data=out_bytes,
                file_name="optimization_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except ValidationError as e:
        st.error(f"Validation failed:\n{e}")
    except Exception as e:
        st.exception(e)
