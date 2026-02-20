# Deployment Recovery Tasks

- [x] Fix Streamlit import compatibility by supporting package and script execution paths in `app/ui.py`.
- [x] Add backward-compatible Excel IO function aliases:
  - `load_tables_from_excel(raw_bytes)`
  - `write_output_excel(...) -> bytes`
- [x] Add backward-compatible orchestrator alias:
  - `run_optimization(tables)`
- [ ] Redeploy Streamlit Cloud app with entrypoint `app/ui.py`.
- [ ] Validate upload/run/download flow with a sample workbook.
- [ ] Confirm generated workbook includes `tblFGResult`, `tblRMDiagnostic`, `tblRunMeta`.
