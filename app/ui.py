import streamlit as st

from .excel_io import load_tables_from_excel, write_output_excel
from .orchestrator import run_optimization
from .validate import ValidationError, validate_inputs


st.set_page_config(page_title="LP Optimizer Service", page_icon="📈", layout="centered")
st.title("LP Optimizer Service for RM-Constrained FG Planning")

upload = st.file_uploader("Upload input Excel (.xlsx)", type=["xlsx"])

if upload is not None:
    raw = upload.read()
    try:
        tables = load_tables_from_excel(raw)
        validate_inputs(tables)
        st.success("Validation passed.")

        if st.button("Run Optimization", type="primary"):
            fg_df, rm_df, meta_df = run_optimization(tables)
            out_bytes = write_output_excel(fg_df, rm_df, meta_df)

            st.subheader("Summary")
            st.write({
                "Total Pairs": int(fg_df["Opt Qty Total"].sum()),
                "Total Margin": float(fg_df["Total Margin"].sum()),
                "RM rows": int(len(rm_df)),
            })

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
