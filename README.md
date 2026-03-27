# LP Optimizer Service for RM-Constrained FG Planning

Streamlit app that uploads an Excel workbook with required input tables, runs a 2-phase FG optimization under RM constraints, and returns an output Excel workbook with result tables.

## Architecture

- `app/ui.py`: Streamlit upload/run/download UI
- `app/excel_io.py`: Excel table extraction, template/sample workbook generation, and output workbook writing
- `app/validate.py`: schema and business-rule validation (fail-fast)
- `app/model.py`: HiGHS-backed MILP solve, LP fallback, and greedy recovery heuristics
- `app/orchestrator.py`: Phase A + strict Phase B gating logic, configurable purchase targets, and explainability metadata

## Solver choice

- **Primary**: MILP using HiGHS via `highspy` (integer production quantities)
- **Fallback**: HiGHS LP relaxation then greedy refill/reallocation; metadata marks heuristic mode and audit warnings when checks fail

## Required input tables

- `fg_master` (or alias `tblFG`)
- `bom_master` (or alias `tblBOM`)
- `tblFGPlanCap`
- `tblRMAvail`
- `tblControl_2`

### Required columns

- `fg_master`: `FG Code` plus either `Margin` or `Unit Margin`
- `bom_master`: `FG Code`, `RM Code`, `QtyPerPair`
- `tblFGPlanCap`: `FG Code` plus either `Max Plan Qty` or `Plan Cap`
- `tblRMAvail`: `RM Code`, `Avail_Stock`, `Avail_StockPO`
- `tblControl_2`: either (`Key`, `Value`), (`Control Key`, `Control Value`), or (`Setting`, `Value`) with keys `Mode_Avail` and `Objective`

The Streamlit UI now includes an input-structure panel and a pre-parse workbook diagnosis step that surfaces table-level issues (missing table objects, blank headers, or empty tables) before validation/optimization.
It also includes download buttons for an input template and a working sample workbook.

## Output workbook

- Sheet `FG_Result`, table `tblFGResult`
  - Includes unmet-cap and likely limiting-RM explainability columns
- Sheet `RM_Diagnostic`, table `tblRMDiagnostic`
  - Includes utilization and binding-RM indicators
- Sheet `Run_Metadata`, table `tblRunMeta`
- Sheet `Purchase_Summary`, table `tblPurchaseSummary`
- Sheet `Purchase_Detail`, table `tblPurchaseDetail` when rows exist
- Additional `Purchase_<target>` sheets are created for each configured purchase target
  - When purchase detail has zero rows, the sheet is exported as header-only without an Excel table object by design to avoid invalid header-only table XML in some Excel readers.

## Install

```bash
pip install -r requirements.txt
```

## Run

Run this command from the project root so the `app` package resolves on `PYTHONPATH`.

```bash
streamlit run app/ui.py
```

For Streamlit Community Cloud, you can also use `streamlit_app.py` as the app entrypoint.

The UI accepts configurable purchase targets as comma-separated percentages and caps default solver threads for safer multi-user execution.

## Test

```bash
pytest -q
```

## Benchmark

Use the synthetic benchmark harness to compare core optimization against purchase-planning runs at larger sizes:

```bash
python scripts/benchmark_solver.py --fg 250 --rm 40 --bom-per-fg 3 --purchase-planner
```

## Python API

`run_optimization` now has a single fixed return contract for all call paths:

```python
fg_result, rm_diag, meta, purchase_summary, purchase_detail = run_optimization(
    tables,
    run_purchase_planner=False,
    purchase_target_fill_pcts="25,50,75,100",
)
```

- `fg_result`: optimized FG quantities and margin output.
- `rm_diag`: RM usage and remaining availability.
- `meta`: run metadata as key/value rows.
- `purchase_summary`: one-row summary of target/achieved plan metrics and buy-cost totals.
- `purchase_detail`: per-RM buy quantities, rate, and buy cost.
