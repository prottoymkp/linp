# FG Optimization App

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run UI

```bash
streamlit run app/ui.py
```

## Input workbook tables

Required named tables:
- `fg_master` or `tblFG`
- `bom_master` or `tblBOM`
- `tblFGPlanCap`
- `tblRMAvail`
- `tblControl_2`

## Output workbook

The app generates an output workbook with:
- `tblFGResult`
- `tblRMDiagnostic`
- `tblRunMeta`

## Tests

```bash
pytest -q
```
