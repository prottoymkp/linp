# LP Optimizer Service for RM-Constrained FG Planning

Streamlit app that uploads an Excel workbook with required input tables, runs a 2-phase FG optimization under RM constraints, and returns an output Excel workbook with result tables.

## Architecture

- `app/ui.py`: Streamlit upload/run/download UI
- `app/excel_io.py`: Excel table extraction and output workbook writing
- `app/validate.py`: schema and business-rule validation (fail-fast)
- `app/model.py`: MILP solve (CBC via PuLP), LP+greedy fallback heuristic
- `app/orchestrator.py`: Phase A + strict Phase B gating logic

## Solver choice

- **Primary**: MILP using PuLP + CBC (integer production quantities)
- **Fallback**: LP relaxation then floor + greedy refill; metadata marks heuristic mode

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
- `tblControl_2`: either (`Key`, `Value`) or (`Control Key`, `Control Value`) with keys `Mode_Avail` and `Objective`

The Streamlit UI now includes an input-structure panel and a pre-parse workbook diagnosis step that surfaces table-level issues (missing table objects, blank headers, or empty tables) before validation/optimization.

## Output workbook

- Sheet `FG_Result`, table `tblFGResult`
- Sheet `RM_Diagnostic`, table `tblRMDiagnostic`
- Sheet `Run_Metadata`, table `tblRunMeta`

## Install

```bash
pip install -r requirements.txt
```

## Run

Run this command from the project root so the `app` package resolves on `PYTHONPATH`.

```bash
streamlit run app/ui.py
```

## Test

```bash
pytest -q
```
