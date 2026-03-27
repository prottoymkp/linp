FG_DATASET = "fg_master"
BOM_DATASET = "bom_master"
CAP_DATASET = "tblFGPlanCap"
RM_DATASET = "tblRMAvail"
CONTROL_DATASET = "tblControl_2"

RM_RATE_COLUMNS = ("RM_Rate", "RM Rate")
PURCHASE_TARGET_CONTROL_KEYS = ("PurchaseTargets", "Purchase_Targets", "Purchase Targets")
DEFAULT_PURCHASE_TARGET_FILL_PCTS = (0.25, 0.50, 0.75, 1.00)
DEFAULT_MAX_SOLVER_THREADS = 4

REQUIRED_TABLES = {
    FG_DATASET: ["FG Code"],
    BOM_DATASET: ["FG Code", "RM Code", "QtyPerPair"],
    CAP_DATASET: ["FG Code"],
    RM_DATASET: ["RM Code", "Avail_Stock", "Avail_StockPO"],
    CONTROL_DATASET: [],
}

TABLE_ALIASES = {
    "tblFG": FG_DATASET,
    "tblBOM": BOM_DATASET,
}

CONTROL_MODE_VALUES = {"STOCK", "STOCK_PO"}
CONTROL_OBJECTIVE_VALUES = {"MARGIN", "PAIRS", "PLAN"}
