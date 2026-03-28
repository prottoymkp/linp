# LP Optimizer Service for RM-Constrained FG Planning

[Open the deployed Streamlit app](https://ah64prvarzspnw5f9ndsbu.streamlit.app/)
[GitHub repository](https://github.com/prottoymkp/linp)

Streamlit optimizer for finished-goods planning, raw-material diagnostics, and buy-cost scenario analysis. Upload an Excel workbook of FG plans, BOM usage, RM availability, and solver controls, then download an output workbook that explains what to build first, which materials are constraining you, and how much extra RM you would need to buy to reach higher fill targets.

![Workflow preview](assets/lp_optimizer_workflow.svg)

## What it does

- Optimizes FG output under RM scarcity using HiGHS-backed mathematical programming.
- Explains bottlenecks through FG and RM diagnostic sheets rather than returning only one final number.
- Supports buy-planning scenarios so planners can compare target fill levels against total RM buy cost.

## Who it is for

- Production planners trying to decide which SKUs to build first under scarce RM.
- Procurement teams estimating the cheapest material buy plan to hit higher production targets.
- Operations or supply-chain analysts who need a workbook-based decision tool without building a custom interface first.

## Business example

A footwear factory has monthly plan caps for 50 FG styles, but leather, outsoles, and trims are limited. Instead of manually juggling spreadsheets, the optimizer picks the best FG mix under current RM limits, shows which materials are binding, and estimates the cheapest buy quantities needed to move from a 50% fill to a 75% or 100% fill target.

## Demo assets

The deployed app exposes both a generated `Sample Input (.xlsx)` download and a generated `Sample Output (.xlsx)` download directly on the page.

This repository keeps the sanitized demo source tables as plain text so the sample stays reviewable and publishable:

- FG demo table: [assets/demo_tables/fg_master.csv](assets/demo_tables/fg_master.csv)
- BOM demo table: [assets/demo_tables/bom_master.csv](assets/demo_tables/bom_master.csv)
- FG cap demo table: [assets/demo_tables/tblFGPlanCap.csv](assets/demo_tables/tblFGPlanCap.csv)
- RM demo table: [assets/demo_tables/tblRMAvail.csv](assets/demo_tables/tblRMAvail.csv)
- Control demo table: [assets/demo_tables/tblControl_2.csv](assets/demo_tables/tblControl_2.csv)
- Workflow preview image: [assets/lp_optimizer_workflow.svg](assets/lp_optimizer_workflow.svg)

The demo tables are a sanitized public sample derived from a real-sized `50 FG / 200 RM` workbook shape. The app rebuilds the downloadable `.xlsx` files from those checked-in tables at runtime.

To materialize local workbook copies from the checked-in demo tables:

```bash
python scripts/build_demo_assets.py --output-dir assets
```

To rebuild the same public shape from a new source workbook first:

```bash
python scripts/build_demo_assets.py --source-workbook "C:\path\to\source.xlsx" --output-dir assets
```

## Required input tables

The app expects named Excel Tables, not plain ranges.

- `fg_master` or alias `tblFG`
- `bom_master` or alias `tblBOM`
- `tblFGPlanCap`
- `tblRMAvail`
- `tblControl_2`

Required columns:

- `fg_master`: `FG Code` plus either `Margin` or `Unit Margin`
- `bom_master`: `FG Code`, `RM Code`, `QtyPerPair`
- `tblFGPlanCap`: `FG Code` plus either `Max Plan Qty` or `Plan Cap`
- `tblRMAvail`: `RM Code`, `Avail_Stock`, `Avail_StockPO`
- `tblControl_2`: key/value columns with at least `Mode_Avail` and `Objective`

## Output workbook walkthrough

### `FG_Result`

This is the production recommendation table. Start here to answer:

- Which FG should we build?
- How much of each plan cap did we actually hit?
- Which FG are constrained by RM?

Key fields:

- `Opt Qty Total`: total recommended production quantity
- `Fill_FG`: share of each FG plan cap achieved
- `Unmet Plan Qty`: how much of the FG target is still not covered
- `Likely Limiting RM`: likely raw material blocking more output
- `Shortfall Reason`: why the target was not fully met

### `RM_Diagnostic`

Use this sheet to validate the bottleneck story from `FG_Result`.

- `availability_total`: RM available to the model
- `availability_used`: RM consumed by the recommended FG plan
- `availability_remaining`: leftover RM after the plan
- `availability_utilization_pct`: percent of availability consumed
- `is_binding`: whether the RM is fully tight and actively constraining the plan

If an FG is short in `FG_Result`, this sheet helps confirm which RM is actually tight across the whole portfolio.

### `Purchase_Summary`

Use this when the base plan is not enough and you need to understand buy scenarios.

- `TargetFillPct`: requested fill threshold for the scenario
- `AchievedPairs`: total FG pairs reached in that scenario
- `AchievedMargin`: resulting total margin
- `TotalBuyCost`: extra RM spend required
- `Status`, `MIPStatus`, `LPStatus`, `Method`: whether the scenario solved exactly or used fallback logic

Interpretation flow:

1. Find under-filled FG in `FG_Result`.
2. Confirm the binding RM in `RM_Diagnostic`.
3. Compare the incremental buy cost in `Purchase_Summary`.

### Other sheets

- `Run_Metadata`: solver settings, timings, fallback flags, and run-level KPIs.
- `Purchase_Detail`: RM-by-RM buy quantities and buy costs for each target scenario.
- `Purchase_<target>`: target-specific FG and RM detail sheets for review or handoff.

## Important planning notes

- Results may be exact or near-optimal depending on `mip_rel_gap`, time limit, and whether fallback heuristics were used.
- Purchase targets are minimum thresholds. The model may exceed a target if that still minimizes the total RM buy cost for the scenario.
- Uploaded workbooks are used only for the active app workflow and are not intentionally stored by the app as part of normal operation. Public demo environments should still use sanitized data.

## What this does not solve yet

- Lead time
- Manpower balancing
- Routing or work-center sequencing
- Cash timing
- Multi-period planning
- Minimum order quantities (MOQ)
- Supplier-specific allocation constraints

## Install

```bash
pip install -r requirements.txt
```

## Run locally

Run this command from the project root so the `app` package resolves on `PYTHONPATH`.

```bash
streamlit run app/ui.py
```

For Streamlit Community Cloud, `streamlit_app.py` is also available as the entrypoint.

## Tests

```bash
pytest -q
```

## Benchmarks

Measured on this environment:

- OS: Windows 11 (`10.0.26200`)
- Python: `3.12.10`
- Solver: HiGHS `1.13.1`
- CPU: `12` logical cores, Intel64 Family 6 Model 186
- App default solver threads: `4`

The numbers below are median wall-clock runtimes across 3 runs and are environment-specific, not universal guarantees.

| Case | Source | Shape | Purchase scenarios | Median runtime |
| --- | --- | --- | --- | --- |
| Small | bundled sample workbook | `50 FG / 200 RM / 400 BOM rows` | `25, 50, 75, 100` | `3.067s` |
| Medium | synthetic harness | `250 FG / 40 RM / 750 BOM rows` | `50, 100` | `81.497s` |
| Heavy | synthetic harness | `1000 FG / 200 RM / 3000 BOM rows` | `50, 100` | `8.834s` |

Benchmark commands:

```bash
python scripts/build_demo_assets.py --output-dir assets
python scripts/benchmark_solver.py --workbook assets/lp_optimizer_sample_input.xlsx --purchase-planner
python scripts/benchmark_solver.py --fg 250 --rm 40 --bom-per-fg 3 --purchase-planner
python scripts/benchmark_solver.py --fg 1000 --rm 200 --bom-per-fg 3 --purchase-planner
```

The medium case is slower than the heavy case because the `250 / 40` synthetic setup is materially denser in RM coupling, which makes the purchase-planning MIP more difficult even though the row counts are smaller.

## Architecture

- [app/ui.py](app/ui.py): Streamlit UI, app-page explainer, and download flow
- [app/excel_io.py](app/excel_io.py): Excel table extraction, template generation, and output workbook writing
- [app/validate.py](app/validate.py): schema and business-rule validation
- [app/model.py](app/model.py): HiGHS-backed optimization and fallback heuristics
- [app/orchestrator.py](app/orchestrator.py): phase logic, purchase scenarios, and explainability metadata
- [scripts/build_demo_assets.py](scripts/build_demo_assets.py): reproducible sample input/output/preview generation
- [scripts/benchmark_solver.py](scripts/benchmark_solver.py): reproducible benchmark harness

## Versioning

- Current in-repo version: `1.0.0`
- Target GitHub repo: `prottoymkp/linp`
- Release prep notes for GitHub About text, topics, and release copy: [docs/release_prep.md](docs/release_prep.md)
- Change summary: [CHANGELOG.md](CHANGELOG.md)
