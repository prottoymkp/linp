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

## Output workbook

- Sheet `FG_Result`, table `tblFGResult`
- Sheet `RM_Diagnostic`, table `tblRMDiagnostic`
- Sheet `Run_Metadata`, table `tblRunMeta`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/ui.py
```

## Test

```bash
pytest -q
```
