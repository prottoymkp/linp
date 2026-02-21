FG_DATASET = "fg_master"
BOM_DATASET = "bom_master"
CAP_DATASET = "tblFGPlanCap"
RM_DATASET = "tblRMAvail"
CONTROL_DATASET = "tblControl_2"

REQUIRED_TABLES = {
    FG_DATASET: ["FG Code"],
    BOM_DATASET: ["FG Code", "RM Code", "QtyPerPair"],
    CAP_DATASET: ["FG Code"],
    RM_DATASET: ["RM Code", "Avail_Stock", "Avail_StockPO", "RM_Rate"],
    CONTROL_DATASET: [],
}

TABLE_ALIASES = {
    "tblFG": FG_DATASET,
    "tblBOM": BOM_DATASET,
}

CONTROL_MODE_VALUES = {"STOCK", "STOCK_PO"}
CONTROL_OBJECTIVE_VALUES = {"MARGIN", "PAIRS", "PLAN"}
