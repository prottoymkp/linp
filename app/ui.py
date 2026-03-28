import base64
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
        st.button(label, disabled=True, help=str(exc), use_container_width=True)
        return
    st.download_button(
        label,
        data=payload,
        file_name=file_name,
        mime=mime,
        use_container_width=True,
    )


def _workflow_preview_data_uri() -> str:
    payload = workflow_preview_svg().encode("utf-8")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _inject_page_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --app-bg: #f5efe6;
                --surface: #fffaf3;
                --surface-muted: #f1e7da;
                --border: #d7c8b8;
                --text-strong: #2f241a;
                --text-body: #5f5246;
                --text-muted: #7b6d60;
                --accent: #6d4528;
                --accent-soft: #efe1cf;
                --teal: #1f5b63;
                --teal-soft: #e8f0ef;
            }

            [data-testid="stAppViewContainer"] {
                background: var(--app-bg);
            }

            [data-testid="stHeader"] {
                background: rgba(245, 239, 230, 0.94);
                border-bottom: 1px solid rgba(215, 200, 184, 0.45);
            }

            .block-container {
                max-width: 1320px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            .block-container [data-testid="stMarkdownContainer"] h1,
            .block-container [data-testid="stMarkdownContainer"] h2,
            .block-container [data-testid="stMarkdownContainer"] h3,
            .block-container [data-testid="stMarkdownContainer"] h4,
            .block-container [data-testid="stMarkdownContainer"] h5,
            .block-container [data-testid="stMarkdownContainer"] h6,
            .block-container label[data-testid="stWidgetLabel"] {
                color: var(--text-strong);
            }

            .block-container [data-testid="stMarkdownContainer"] p,
            .block-container [data-testid="stMarkdownContainer"] li,
            .block-container [data-testid="stCaptionContainer"],
            .block-container div[data-testid="stMetricValue"],
            .block-container div[data-testid="stMetricLabel"] {
                color: var(--text-body);
            }

            .hero-shell {
                margin-bottom: 1.2rem;
                padding: 1.7rem 1.8rem;
                border-radius: 28px;
                border: 1px solid var(--border);
                background: var(--surface);
                box-shadow: 0 12px 24px rgba(82, 61, 42, 0.05);
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: var(--teal-soft);
                border: 1px solid rgba(31, 91, 99, 0.16);
                color: var(--teal);
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            .hero-shell h1 {
                margin: 0.8rem 0 0.45rem;
                color: var(--text-strong);
                font-size: clamp(2rem, 4vw, 3.35rem);
                line-height: 1.05;
                letter-spacing: -0.03em;
            }

            .hero-copy {
                max-width: 58rem;
                margin: 0;
                color: var(--text-body);
                font-size: 1.03rem;
                line-height: 1.55;
            }

            .hero-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.85rem;
                margin-top: 1.25rem;
            }

            .hero-card {
                padding: 1rem 1.05rem;
                border-radius: 22px;
                background: var(--surface-muted);
                border: 1px solid var(--border);
            }

            .hero-card-label {
                display: block;
                color: var(--teal);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .hero-card strong {
                display: block;
                margin-top: 0.25rem;
                color: var(--text-strong);
                font-size: 1.02rem;
            }

            .hero-card p {
                margin: 0.38rem 0 0;
                color: var(--text-body);
                font-size: 0.92rem;
                line-height: 1.45;
            }

            .section-label {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                margin: 0 0 0.75rem;
                padding: 0.34rem 0.7rem;
                border-radius: 999px;
                background: var(--accent-soft);
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            .mini-steps {
                display: grid;
                gap: 0.7rem;
                margin-bottom: 1rem;
            }

            .mini-step {
                padding: 0.85rem 0.95rem;
                border-radius: 18px;
                background: var(--surface-muted);
                border: 1px solid var(--border);
            }

            .mini-step strong {
                display: block;
                color: var(--text-strong);
                font-size: 0.95rem;
            }

            .mini-step span {
                display: block;
                margin-top: 0.2rem;
                color: var(--text-body);
                font-size: 0.9rem;
                line-height: 1.4;
            }

            .workflow-zoom-link {
                display: block;
                text-decoration: none;
            }

            .workflow-zoom-frame {
                position: relative;
                overflow: hidden;
                border-radius: 18px;
                border: 1px solid var(--border);
                background: var(--surface);
            }

            .workflow-zoom-frame img {
                display: block;
                width: 100%;
                height: auto;
                cursor: zoom-in;
                transition: transform 180ms ease;
            }

            .workflow-zoom-frame:hover img {
                transform: scale(1.015);
            }

            .workflow-zoom-badge {
                position: absolute;
                top: 0.85rem;
                right: 0.85rem;
                padding: 0.35rem 0.6rem;
                border-radius: 999px;
                background: var(--accent);
                color: #ffffff;
                font-size: 0.74rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                box-shadow: none;
            }

            .workflow-zoom-modal {
                position: fixed;
                inset: 0;
                display: none;
                align-items: center;
                justify-content: center;
                padding: 2rem;
                background: rgba(9, 25, 37, 0.8);
                z-index: 1000;
            }

            .workflow-zoom-modal:target {
                display: flex;
            }

            .workflow-zoom-backdrop {
                position: absolute;
                inset: 0;
                display: block;
            }

            .workflow-zoom-dialog {
                position: relative;
                z-index: 1;
                width: min(94vw, 1460px);
                max-height: 90vh;
                padding: 1rem 1rem 0.8rem;
                border-radius: 24px;
                border: 1px solid var(--border);
                background: var(--surface);
                box-shadow: 0 24px 48px rgba(47, 36, 26, 0.18);
            }

            .workflow-zoom-dialog img {
                display: block;
                width: 100%;
                height: auto;
                max-height: calc(90vh - 5.25rem);
                object-fit: contain;
                border-radius: 16px;
                background: #ffffff;
            }

            .workflow-zoom-close {
                position: absolute;
                top: 0.9rem;
                right: 0.9rem;
                padding: 0.42rem 0.82rem;
                border-radius: 999px;
                background: var(--accent);
                color: #ffffff;
                font-size: 0.8rem;
                font-weight: 700;
                text-decoration: none;
            }

            .workflow-zoom-dialog-caption {
                margin: 0.72rem 0 0;
                color: var(--text-muted);
                font-size: 0.92rem;
                text-align: center;
            }

            div[data-testid="stFileUploader"] {
                border-radius: 20px;
                border: 1.5px dashed rgba(109, 69, 40, 0.35);
                background: var(--surface);
                padding: 0.35rem 0.45rem;
            }

            div[data-testid="stFileUploader"] section {
                padding: 0.15rem 0.25rem;
            }

            div[data-testid="stFileUploader"] * {
                color: var(--text-body);
            }

            div[data-testid="stButton"] > button,
            div[data-testid="stDownloadButton"] > button {
                border-radius: 999px;
                min-height: 2.8rem;
                font-weight: 700;
                box-shadow: none;
            }

            div[data-testid="stDownloadButton"] > button {
                background: var(--surface);
                color: var(--accent);
                border: 1px solid rgba(109, 69, 40, 0.24);
            }

            div[data-testid="stDownloadButton"] > button:hover {
                border-color: rgba(109, 69, 40, 0.42);
                color: var(--accent);
                background: var(--accent-soft);
            }

            div[data-testid="stButton"] > button[kind="primary"] {
                background: var(--accent);
                color: #ffffff;
                border: 1px solid var(--accent);
            }

            div[data-testid="stButton"] > button[kind="primary"]:hover {
                background: #5c3820;
                color: #ffffff;
                border-color: #5c3820;
            }

            div[data-testid="stMetric"] {
                padding: 0.5rem 0.75rem;
                border-radius: 18px;
                background: var(--surface-muted);
                border: 1px solid var(--border);
            }

            div[data-testid="stExpander"] {
                border-radius: 18px;
                border: 1px solid var(--border);
                background: var(--surface);
            }

            div[data-testid="stAlert"] {
                border-radius: 18px;
                border: 1px solid var(--border);
            }

            div[data-testid="stAlert"][kind="info"] {
                background: #edf3fb;
                color: var(--text-body);
            }

            div[data-testid="stAlert"][kind="success"] {
                background: #ecf4ed;
                color: var(--text-body);
            }

            div[data-testid="stAlert"][kind="warning"] {
                background: #fbf1df;
                color: var(--text-body);
            }

            div[data-testid="stAlert"][kind="error"] {
                background: #fbe8e6;
                color: var(--text-body);
            }

            @media (max-width: 960px) {
                .hero-grid {
                    grid-template-columns: 1fr;
                }

                .block-container {
                    padding-top: 1.2rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-badge">LP Optimizer - v{__version__}</div>
            <h1>RM Planning Optimizer</h1>
            <p class="hero-copy">
                RM-Constrained FG Planning.
                {APP_DESCRIPTION}
                The page is organized so the working area comes first: upload a workbook,
                validate the structure, tune the solver, and download the optimized output
                before diving into the narrative details below.
            </p>
            <div class="hero-grid">
                <div class="hero-card">
                    <span class="hero-card-label">Step 1</span>
                    <strong>Upload the workbook first</strong>
                    <p>The input drop zone now sits at the top so users can start immediately.</p>
                </div>
                <div class="hero-card">
                    <span class="hero-card-label">Step 2</span>
                    <strong>Keep controls in view</strong>
                    <p>Validation, purchase planning, and solver settings stay close to the upload flow.</p>
                </div>
                <div class="hero-card">
                    <span class="hero-card-label">Step 3</span>
                    <strong>Read the narrative later</strong>
                    <p>The business context and output guide move into a lower section for cleaner focus.</p>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_requirements_list() -> None:
    st.caption("Accepted aliases: `tblFG` -> `fg_master`, `tblBOM` -> `bom_master`")
    for table_name, required_columns in REQUIRED_TABLES.items():
        columns = ", ".join(required_columns) if required_columns else "No mandatory columns, but key/value control rows are required."
        st.write(f"- **{table_name}**: {columns}")
    st.info("Optional accepted columns: `fg_master` accepts `Margin` or `Unit Margin`; `tblFGPlanCap` accepts `Max Plan Qty` or `Plan Cap`.")


def _render_starter_kit() -> None:
    with st.container(border=True):
        st.markdown("### Starter kit")
        st.caption("Grab a workbook, inspect the sample output, or hand teammates a clean template.")
        download_left, download_right = st.columns(2)
        with download_left:
            st.download_button(
                "Template (.xlsx)",
                data=build_input_workbook(include_sample_rows=False),
                file_name="lp_optimizer_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with download_right:
            _render_generated_download(
                "Sample Input (.xlsx)",
                sample_input_bytes,
                "lp_optimizer_sample_input.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        _render_generated_download(
            "Sample Output (.xlsx)",
            sample_output_bytes,
            "lp_optimizer_sample_output.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def _render_workbook_map() -> None:
    with st.container(border=True):
        st.markdown("### Workbook map")
        st.caption("Use named Excel Tables, not plain ranges.")
        _render_requirements_list()


def _render_workflow_glance() -> None:
    with st.container(border=True):
        st.markdown("### Workflow glance")
        st.markdown(
            """
            <div class="mini-steps">
                <div class="mini-step">
                    <strong>Prepare</strong>
                    <span>Start from the template or sample workbook so the table names are already right.</span>
                </div>
                <div class="mini-step">
                    <strong>Run</strong>
                    <span>Upload the workbook, fix any structural issues, and launch the optimizer from the left panel.</span>
                </div>
                <div class="mini-step">
                    <strong>Review</strong>
                    <span>Download the output workbook and inspect FG, RM, and purchase-planning sheets.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview_uri = _workflow_preview_data_uri()
        st.markdown(
            f"""
            <div id="workflow-preview-panel">
                <a class="workflow-zoom-link" href="#workflow-preview-modal" title="Zoom workflow preview">
                    <div class="workflow-zoom-frame">
                        <span class="workflow-zoom-badge">Click to zoom</span>
                        <img src="{preview_uri}" alt="LP Optimizer workflow preview" />
                    </div>
                </a>
            </div>
            <div id="workflow-preview-modal" class="workflow-zoom-modal" role="dialog" aria-modal="true" aria-label="Workflow preview">
                <a class="workflow-zoom-backdrop" href="#workflow-preview-panel" aria-label="Close workflow preview"></a>
                <div class="workflow-zoom-dialog">
                    <a class="workflow-zoom-close" href="#workflow-preview-panel" aria-label="Close workflow preview">Close</a>
                    <img src="{preview_uri}" alt="LP Optimizer workflow preview" />
                    <p class="workflow-zoom-dialog-caption">Zoom your browser if you want to inspect the fine-grained sheet labels.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Click the workflow preview to open a larger in-page view.")


def _render_bottom_guide() -> None:
    st.markdown('<div class="section-label">Planner guide</div>', unsafe_allow_html=True)
    st.caption("Narrative context and interpretation live below the control deck so the workflow stays above the fold.")

    top_left, top_right = st.columns(2, gap="large")
    with top_left:
        with st.container(border=True):
            st.markdown("### What it does")
            st.write(
                """
                This app turns a workbook of finished-goods plans, raw-material availability,
                BOM usage, and planner controls into an optimized production plan. It helps teams
                see which FG quantities are worth building first, which RMs are binding, and how
                much extra material they would need to buy to hit higher fill targets.
                """
            )

        with st.container(border=True):
            st.markdown("### Business example")
            st.write(
                """
                Imagine a footwear factory that has monthly FG targets but not enough leather,
                soles, or trims to make everything. Instead of manually comparing dozens of SKUs
                and hundreds of materials, the optimizer recommends the best production mix under
                current RM constraints and then shows the lowest-cost buy plan for hitting higher
                service levels.
                """
            )

    with top_right:
        with st.container(border=True):
            st.markdown("### Who it is for")
            st.write(
                """
                It is designed for production planners, procurement teams, operations leaders,
                and analysts who need a fast answer to: "Given scarce RM, what should we build
                now, and what would it cost to buy our way to a higher plan fill?"
                """
            )

        with st.container(border=True):
            st.markdown("### What each output sheet means")
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

    with st.container(border=True):
        st.markdown("### Run notes")
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

        detail_left, detail_right = st.columns(2, gap="large")
        with detail_left:
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
        with detail_right:
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


st.set_page_config(page_title="LP Optimizer Service", page_icon=":bar_chart:", layout="wide")

_inject_page_styles()
_render_hero()

st.markdown('<div class="section-label">Control deck</div>', unsafe_allow_html=True)

main_col, side_col = st.columns([1.35, 0.95], gap="large")

with main_col:
    with st.container(border=True):
        st.markdown("### Upload workbook")
        st.caption("Start here. Once the workbook checks out, the optimization controls unlock directly below.")
        upload = st.file_uploader("Upload input Excel (.xlsx)", type=["xlsx"], label_visibility="collapsed")
        if upload is None:
            st.info("Upload an `.xlsx` workbook to unlock validation, solver controls, and the output download.")

with side_col:
    _render_starter_kit()

if upload is None:
    with main_col:
        with st.container(border=True):
            st.markdown("### What happens next")
            st.markdown(
                """
                - The app checks the workbook structure and reports missing or mismatched tables.
                - Once validation passes, purchase-planning and solver settings become available in the same column.
                - After the run finishes, the output workbook download and summary stay right below the controls.
                """
            )
else:
    raw = upload.read()
    try:
        diagnosis = diagnose_workbook_structure(raw)

        with main_col:
            with st.container(border=True):
                st.markdown("### File readiness")
                status_left, status_mid, status_right = st.columns(3)
                with status_left:
                    st.metric("Detected tables", f"{len(diagnosis['tables'])}")
                with status_mid:
                    st.metric("Warnings", f"{len(diagnosis['warnings'])}")
                with status_right:
                    st.metric("Issues", f"{len(diagnosis['issues'])}")

                st.caption(f"Current workbook: `{upload.name}` - {len(raw) / 1024:.1f} KB")

                if diagnosis["issues"]:
                    for issue in diagnosis["issues"]:
                        st.error(issue)
                if diagnosis["warnings"]:
                    for warning in diagnosis["warnings"]:
                        st.warning(warning)
                if diagnosis["tables"]:
                    detected_tables = ", ".join(f"`{table_name}`" for table_name in diagnosis["tables"])
                    st.markdown(f"Detected tables: {detected_tables}")

        if diagnosis["issues"]:
            with main_col:
                with st.container(border=True):
                    st.markdown("### Optimization controls")
                    st.info("Fix the workbook issues above and re-upload. Controls stay disabled until the structure checks pass.")
        else:
            tables = load_tables_from_excel(raw)
            validate_inputs(tables)

            with main_col:
                with st.container(border=True):
                    st.markdown("### Optimization controls")
                    st.success("Validation passed. The workbook is ready to run.")

                    control_left, control_right = st.columns(2, gap="large")
                    with control_left:
                        run_purchase_planner = st.checkbox("Run purchase planner", value=True)
                        purchase_targets_raw = "25,50,75,100"
                        if run_purchase_planner:
                            purchase_targets_raw = st.text_input(
                                "Purchase targets (%)",
                                value="25,50,75,100",
                                help="Comma-separated minimum fill targets like 25,50,75,100.",
                            )
                        else:
                            st.caption("Purchase target scenarios are skipped when purchase planning is off.")

                    with control_right:
                        threads_raw = st.text_input("threads (integer, blank = auto)", value="")
                        mip_rel_gap_raw = st.text_input("mip_rel_gap (float, default 0.01)", value="0.01")
                        time_limit_sec_raw = st.text_input("time_limit_sec (float, blank = unset)", value="")

                    st.caption("Blank `threads` uses the app default and is capped for safer multi-user execution.")
                    run_clicked = st.button("Run optimization", type="primary", use_container_width=True)

            if run_clicked:
                threads_value = _optional_int_value(threads_raw, "threads")
                mip_rel_gap_value = _optional_float_value(mip_rel_gap_raw, "mip_rel_gap", min_value=0.0, max_value=1.0)
                time_limit_value = _optional_float_value(time_limit_sec_raw, "time_limit_sec", min_value=0.0, min_inclusive=False)
                purchase_targets_value = _purchase_targets_value(purchase_targets_raw) if run_purchase_planner else None

                with main_col:
                    with st.container(border=True):
                        st.markdown("### Solver progress")

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
                    with main_col:
                        with st.container(border=True):
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

                with main_col:
                    with st.container(border=True):
                        st.markdown("### Results")
                        summary_left, summary_mid, summary_right = st.columns(3)
                        with summary_left:
                            st.metric("Total pairs", f"{int(fg_df['Opt Qty Total'].sum()):,}")
                        with summary_mid:
                            st.metric("Total margin", f"{float(fg_df['Total Margin'].sum()):,.2f}")
                        with summary_right:
                            st.metric("RM rows", f"{int(len(rm_df)):,}")

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
                            with st.expander("Purchase summary preview", expanded=True):
                                st.dataframe(purchase_summary_df, use_container_width=True)

                        with st.expander("Run metadata", expanded=False):
                            st.dataframe(meta_df, use_container_width=True)

                        st.download_button(
                            "Download optimized output (.xlsx)",
                            data=out_bytes,
                            file_name="optimization_output.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
    except ValidationError as exc:
        with main_col:
            with st.container(border=True):
                st.error(f"Validation failed:\n{exc}")
    except UserInputError as exc:
        with main_col:
            with st.container(border=True):
                st.error(f"Input error:\n{exc}")
    except Exception as exc:
        with main_col:
            with st.container(border=True):
                st.exception(exc)

support_left, support_right = st.columns([1.02, 0.98], gap="large")

with support_left:
    _render_workbook_map()

with support_right:
    _render_workflow_glance()

st.divider()
_render_bottom_guide()
