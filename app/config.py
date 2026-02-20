"""Application-wide constants for workbook schema and controls."""

FG_DATASET = "fg_master"
BOM_DATASET = "bom_master"
CAP_DATASET = "fg_plan_cap"
RM_DATASET = "rm_avail"
CONTROL_DATASET = "control"

TABLE_ALIASES = {
    "fg_master": FG_DATASET,
    "tblFG": FG_DATASET,
    "bom_master": BOM_DATASET,
    "tblBOM": BOM_DATASET,
    "tblFGPlanCap": CAP_DATASET,
    "tblRMAvail": RM_DATASET,
    "tblControl_2": CONTROL_DATASET,
}

REQUIRED_TABLES = {
    FG_DATASET,
    BOM_DATASET,
    CAP_DATASET,
    RM_DATASET,
    CONTROL_DATASET,
}

REQUIRED_COLUMNS = {
    FG_DATASET: ["FG Code", "Unit Margin"],
    BOM_DATASET: ["FG Code", "RM Code", "QtyPerPair"],
    CAP_DATASET: ["FG Code", "Plan Cap"],
    RM_DATASET: ["RM Code", "Avail_Stock", "Avail_StockPO"],
    CONTROL_DATASET: ["Control Key", "Control Value"],
}

CONTROL_KEYS = {
    "Mode_Avail": {"STOCK", "STOCK_PO"},
    "Objective": {"MARGIN", "PAIRS"},
}

DEFAULT_PHASE_B_UPPER_BOUND = 1_000_000
