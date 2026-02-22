from pathlib import Path
import sys
import time

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import REQUIRED_TABLES
from app.excel_io import diagnose_workbook_structure, load_tables_from_excel, write_output_excel
from app.orchestrator import run_optimization
from app.validate import ValidationError, validate_inputs


def _optional_int_value(raw: str) -> int | None:
    text = str(raw).strip()
    if not text:
        return None
    return int(text)


def _optional_float_value(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    return float(text)

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

        with st.expander("Advanced solver controls", expanded=False):
            threads_raw = st.text_input("threads (integer, blank = auto)", value="")
            mip_rel_gap_raw = st.text_input("mip_rel_gap (float, default 0.01)", value="0.01")
            time_limit_sec_raw = st.text_input("time_limit_sec (float, blank = unset)", value="")

        if st.button("Run Optimization", type="primary"):
            stage_holder = st.empty()
            overall_holder = st.empty()
            elapsed_holder = st.empty()
            heartbeat_holder = st.empty()
            stage_progress = st.progress(0, text="Current stage progress: 0%")
            overall_progress = st.progress(0, text="Overall progress: 0%")
            run_start = time.monotonic()

            def on_progress(stage: str, stage_pct: float, overall_pct: float, status_text: str, is_heartbeat: bool) -> None:
                elapsed_total = time.monotonic() - run_start
                stage_holder.info(f"Current stage: {stage}")
                overall_holder.caption(f"Solver overall progress: {overall_pct:.0f}%")
                elapsed_holder.caption(f"Elapsed runtime: {elapsed_total:.1f}s")
                if is_heartbeat:
                    heartbeat_holder.warning(f"Solver still running… {status_text}")
                else:
                    heartbeat_holder.info(status_text)
                stage_progress.progress(int(stage_pct), text=f"Current stage progress: {stage_pct:.0f}%")
                overall_progress.progress(int(overall_pct), text=f"Overall progress: {overall_pct:.0f}%")

            fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
                tables,
                run_purchase_planner=run_purchase_planner,
                progress_callback=on_progress,
                threads=_optional_int_value(threads_raw),
                mip_rel_gap=_optional_float_value(mip_rel_gap_raw),
                time_limit_sec=_optional_float_value(time_limit_sec_raw),
            )
            meta_map = dict(zip(meta_df["Key"], meta_df["Value"])) if {"Key", "Value"}.issubset(meta_df.columns) else {}
            if str(meta_map.get("heuristic_cutoff_hit", "False")).lower() == "true":
                st.warning(
                    "Phase A fallback heuristic ended due to safety limits "
                    f"(reason: {meta_map.get('cutoff_reason', 'unknown')}, "
                    f"iterations: {meta_map.get('heuristic_iterations', 'n/a')}, "
                    f"elapsed sec: {meta_map.get('fallback_elapsed_sec', 'n/a')})."
                )
            purchase_target_sheets = purchase_detail_df.attrs.get("purchase_target_sheets")
            try:
                out_bytes = write_output_excel(
                    fg_df,
                    rm_df,
                    meta_df,
                    purchase_summary_df,
                    purchase_detail_df,
                    purchase_target_sheets=purchase_target_sheets,
                )
            except TypeError as exc:
                if "purchase_target_sheets" not in str(exc):
                    raise
                out_bytes = write_output_excel(
                    fg_df,
                    rm_df,
                    meta_df,
                    purchase_summary_df,
                    purchase_detail_df,
                )

            st.subheader("Summary")
            st.write({
                "Total Pairs": int(fg_df["Opt Qty Total"].sum()),
                "Total Margin": float(fg_df["Total Margin"].sum()),
                "RM rows": int(len(rm_df)),
            })

            if run_purchase_planner and not purchase_summary_df.empty and "Status" in purchase_summary_df.columns:
                fallback_rows = purchase_summary_df[purchase_summary_df["Status"].astype(str).str.startswith("fallback_")]
                if not fallback_rows.empty:
                    st.info(
                        "Purchase planner fallback statuses detected: "
                        + ", ".join(sorted(fallback_rows["Status"].astype(str).unique()))
                        + ". Check MIPStatus/LPStatus and cutoff columns for diagnostics."
                    )

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
