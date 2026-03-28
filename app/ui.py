from pathlib import Path
import sys
import time

import streamlit as st
from streamlit.errors import NoSessionContext


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import APP_DESCRIPTION, __version__
from app.assets import sample_input_bytes, sample_output_bytes, workflow_preview_svg
from app.config import REQUIRED_TABLES
from app.excel_io import build_input_workbook, diagnose_workbook_structure, load_tables_from_excel, write_output_excel
from app.orchestrator import run_optimization
from app.validate import ValidationError, validate_inputs


class UserInputError(Exception):
    pass


def _optional_int_value(raw: str, field_name: str) -> int | None:
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() in {"auto", "default", "none", "null", "unset"}:
        return None
    try:
        value = int(float(text))
    except ValueError as exc:
        raise UserInputError(f"{field_name} must be an integer or left blank.") from exc
    if value < 1:
        raise UserInputError(f"{field_name} must be at least 1.")
    return value


def _optional_float_value(
    raw: str,
    field_name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() in {"default", "none", "null", "unset"}:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise UserInputError(f"{field_name} must be a number or left blank.") from exc
    if min_value is not None:
        too_low = value < min_value if min_inclusive else value <= min_value
        if too_low:
            comparator = "at least" if min_inclusive else "greater than"
            raise UserInputError(f"{field_name} must be {comparator} {min_value}.")
    if max_value is not None and value > max_value:
        raise UserInputError(f"{field_name} must be at most {max_value}.")
    return value


def _purchase_targets_value(raw: str) -> str | None:
    text = str(raw).strip()
    if not text:
        raise UserInputError("Purchase targets cannot be blank when purchase planning is enabled.")
    tokens = [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]
    if not tokens:
        raise UserInputError("Purchase targets must contain at least one percentage.")
    for token in tokens:
        try:
            value = float(token)
        except ValueError as exc:
            raise UserInputError("Purchase targets must be comma-separated numbers like 25,50,75,100.") from exc
        normalized = value / 100.0 if value > 1.0 else value
        if normalized <= 0 or normalized > 1.0:
            raise UserInputError("Purchase targets must be between 0 and 100 percent.")
    return ",".join(tokens)


def _render_generated_download(label: str, data_factory, file_name: str, mime: str) -> None:
    try:
        payload = data_factory()
    except FileNotFoundError as exc:
        st.button(label, disabled=True, help=str(exc))
        return
    st.download_button(
        label,
        data=payload,
        file_name=file_name,
        mime=mime,
    )


def _render_intro() -> None:
    st.caption(f"Version {__version__}")
    st.write(APP_DESCRIPTION)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            """
            ### What it does
            This app turns a workbook of finished-goods plans, raw-material availability,
            BOM usage, and planner controls into an optimized production plan. It shows
            which FG quantities are worth building first, which RMs are binding, and how
            much extra material you would need to buy to hit higher fill targets.
            """
        )
    with right_col:
        st.markdown(
            """
            ### Who it is for
            It is designed for production planners, procurement teams, operations leaders,
            and analysts who need a fast answer to: "Given scarce RM, what should we build
            now, and what would it cost to buy our way to a higher plan fill?"
            """
        )

    st.markdown(
        """
        ### Business example
        Imagine a footwear factory that has monthly FG targets but not enough leather,
        soles, or trims to make everything. Instead of manually comparing dozens of SKUs
        and hundreds of materials, the optimizer recommends the best production mix under
        current RM constraints and then shows the lowest-cost buy plan for hitting higher
        service levels.
        """
    )

    st.markdown(workflow_preview_svg(), unsafe_allow_html=True)
    st.caption("Quick workflow preview: download a sample input, run the optimizer, then review the output workbook.")

    with st.expander("What each output sheet means", expanded=True):
        st.markdown(
            """
            - `FG_Result`: final quantity recommendation by FG, including phase split, fill rate, unmet plan quantity, and likely limiting RM.
            - `RM_Diagnostic`: RM availability, usage, remaining balance, utilization percent, and whether an RM is fully binding.
            - `Purchase_Summary`: target-by-target buy scenarios showing achieved pairs, achieved margin, total buy cost, and solver status.
            - `Run_Metadata`: solver settings, solver path, fallback signals, and run-level KPIs for auditability.
            - `Purchase_Detail`: RM-level buy quantities and buy cost lines for the active purchase scenarios.
            - `Purchase_<target>`: target-specific FG and RM detail sheets so planners can inspect one fill threshold at a time.
            """
        )

    note_left, note_right = st.columns(2)
    with note_left:
        st.info(
            "Results may be exact or near-optimal depending on solver settings, including "
            "`mip_rel_gap`, `time_limit_sec`, and whether fallback heuristics were needed."
        )
    with note_right:
        st.info(
            "Purchase targets are minimum thresholds. The model may exceed a target if doing "
            "so still minimizes the total RM buy cost for that scenario."
        )

    with st.expander("What this does not solve yet", expanded=False):
        st.markdown(
            """
            - Lead-time-aware material availability
            - Manpower or line-capacity balancing
            - Routing or work-center sequencing
            - Cash timing and working-capital constraints
            - Multi-period planning across weeks or months
            - Minimum order quantities (MOQ)
            - Supplier allocation or vendor-specific constraints
            """
        )

    with st.expander("Privacy note", expanded=False):
        st.markdown(
            """
            Uploaded workbooks are used only for the active app session to validate inputs,
            run the optimization, and prepare the downloadable output workbook. This app does
            not intentionally persist uploaded files as part of the normal workflow, but you
            should still avoid sharing confidential data in public demo environments unless it
            has already been sanitized.
            """
        )


def _render_input_requirements() -> None:
    with st.expander("Input file structure requirements", expanded=True):
        st.markdown("Upload an **.xlsx workbook** that contains named Excel Tables with the following structures.")
        st.caption("Table aliases accepted: `tblFG` -> `fg_master`, `tblBOM` -> `bom_master`")
        template_col, sample_input_col, sample_output_col = st.columns(3)
        with template_col:
            st.download_button(
                "Template (.xlsx)",
                data=build_input_workbook(include_sample_rows=False),
                file_name="lp_optimizer_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with sample_input_col:
            _render_generated_download(
                "Sample Input (.xlsx)",
                sample_input_bytes,
                "lp_optimizer_sample_input.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with sample_output_col:
            _render_generated_download(
                "Sample Output (.xlsx)",
                sample_output_bytes,
                "lp_optimizer_sample_output.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        for table_name, required_columns in REQUIRED_TABLES.items():
            cols = ", ".join(required_columns) if required_columns else "No mandatory columns, but must include control key/value pairs."
            st.write(f"- **{table_name}**: {cols}")

        st.info("Optional accepted columns: `fg_master` accepts `Margin` or `Unit Margin`; `tblFGPlanCap` accepts `Max Plan Qty` or `Plan Cap`.")


def _run_quality_message(meta_map: dict[str, str]) -> str:
    phase_status = str(meta_map.get("phase_a_status", "unknown"))
    purchase_status = str(meta_map.get("purchase_plan_status", "not_run"))
    heuristic_cutoff = str(meta_map.get("heuristic_cutoff_hit", "False")).lower() == "true"
    if phase_status == "Optimal" and not heuristic_cutoff and purchase_status not in {"fallback_deficit_buy", "skipped_missing_rm_rate"}:
        return "This run reached exact optimal solutions for the executed models under the current settings."
    return (
        "This run includes relaxed or fallback behavior, so treat the result as near-optimal "
        "and review `Run_Metadata`, `Purchase_Summary`, and any warnings before operationalizing it."
    )


st.set_page_config(page_title="LP Optimizer Service", page_icon=":bar_chart:", layout="centered")
st.title("LP Optimizer Service for RM-Constrained FG Planning")

_render_intro()
_render_input_requirements()

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

        run_purchase_planner = st.checkbox("Run purchase planner", value=True)
        purchase_targets_raw = "25,50,75,100"
        if run_purchase_planner:
            purchase_targets_raw = st.text_input("purchase targets (%)", value="25,50,75,100")

        with st.expander("Advanced solver controls", expanded=False):
            threads_raw = st.text_input("threads (integer, blank = auto)", value="")
            mip_rel_gap_raw = st.text_input("mip_rel_gap (float, default 0.01)", value="0.01")
            time_limit_sec_raw = st.text_input("time_limit_sec (float, blank = unset)", value="")
            st.caption("Blank threads uses the app default and is capped for safer multi-user execution.")

        if st.button("Run Optimization", type="primary"):
            threads_value = _optional_int_value(threads_raw, "threads")
            mip_rel_gap_value = _optional_float_value(mip_rel_gap_raw, "mip_rel_gap", min_value=0.0, max_value=1.0)
            time_limit_value = _optional_float_value(time_limit_sec_raw, "time_limit_sec", min_value=0.0, min_inclusive=False)
            purchase_targets_value = _purchase_targets_value(purchase_targets_raw) if run_purchase_planner else None

            stage_holder = st.empty()
            overall_holder = st.empty()
            elapsed_holder = st.empty()
            heartbeat_holder = st.empty()
            stage_progress = st.progress(0, text="Current stage progress: 0%")
            overall_progress = st.progress(0, text="Overall progress: 0%")
            run_start = time.monotonic()

            def on_progress(stage: str, stage_pct: float, overall_pct: float, status_text: str, is_heartbeat: bool) -> None:
                try:
                    elapsed_total = time.monotonic() - run_start
                    stage_holder.info(f"Current stage: {stage}")
                    overall_holder.caption(f"Solver overall progress: {overall_pct:.0f}%")
                    elapsed_holder.caption(f"Elapsed runtime: {elapsed_total:.1f}s")
                    if is_heartbeat:
                        heartbeat_holder.warning(f"Solver still running... {status_text}")
                    else:
                        heartbeat_holder.info(status_text)
                    stage_progress.progress(int(stage_pct), text=f"Current stage progress: {stage_pct:.0f}%")
                    overall_progress.progress(int(overall_pct), text=f"Overall progress: {overall_pct:.0f}%")
                except NoSessionContext:
                    # Streamlit widgets cannot be updated from detached background threads
                    # after the client disconnects; ignore these stale heartbeat updates.
                    return

            fg_df, rm_df, meta_df, purchase_summary_df, purchase_detail_df = run_optimization(
                tables,
                run_purchase_planner=run_purchase_planner,
                progress_callback=on_progress,
                threads=threads_value,
                mip_rel_gap=mip_rel_gap_value,
                time_limit_sec=time_limit_value,
                purchase_target_fill_pcts=purchase_targets_value,
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
            st.write(
                {
                    "Total Pairs": int(fg_df["Opt Qty Total"].sum()),
                    "Total Margin": float(fg_df["Total Margin"].sum()),
                    "RM rows": int(len(rm_df)),
                }
            )
            st.caption(_run_quality_message(meta_map))

            if run_purchase_planner and not purchase_summary_df.empty and "Status" in purchase_summary_df.columns:
                fallback_rows = purchase_summary_df[purchase_summary_df["Status"].astype(str).str.startswith("fallback_")]
                if not fallback_rows.empty:
                    st.info(
                        "Purchase planner fallback statuses detected: "
                        + ", ".join(sorted(fallback_rows["Status"].astype(str).unique()))
                        + ". Check `MIPStatus`, `LPStatus`, and cutoff columns for diagnostics."
                    )
                audit_warning_rows = (
                    purchase_summary_df[purchase_summary_df.get("AuditWarning", "none").astype(str).str.lower() != "none"]
                    if "AuditWarning" in purchase_summary_df.columns
                    else purchase_summary_df.iloc[0:0]
                )
                if not audit_warning_rows.empty:
                    st.warning("One or more purchase-plan audit checks reported warnings. Review the `AuditWarning` column in `Purchase_Summary`.")

            if run_purchase_planner and not purchase_summary_df.empty:
                st.subheader("Purchase summary preview")
                st.dataframe(purchase_summary_df, use_container_width=True)

            st.download_button(
                "Download optimized output (.xlsx)",
                data=out_bytes,
                file_name="optimization_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except ValidationError as e:
        st.error(f"Validation failed:\n{e}")
    except UserInputError as e:
        st.error(f"Input error:\n{e}")
    except Exception as e:
        st.exception(e)
