from enum import Enum
from importlib.resources import files


class ModelType(Enum):
    # Baseline
    REG = "reg"
    # BIN-BASED
    BIN = "bin"  # bin number optimized by optuna
    BIN_N = "bin_n"  # binning with fixed number of bins
    # SIG-BASED
    SIG = "sig"
    DSIG = "dsig"
    # BOX-BASED
    BOX = "box"  # order optimized by loss
    BOX_ORD = "box_ord"  # order optimized by optuna
    BOX_ORD_N = "box_ord_n"  # box with fixed order
    CDF = "cdf"
    HAZ_STEP = "haz_step"
    # EXTERNAL BASELINES
    # pycox
    BCE_SURV = "bce_surv"
    COX = "cox"
    COX_CC = "cox_cc"
    COX_PH = "cox_ph"
    DEEP_HIT = "deep_hit"
    DSM = "dsm"
    LOG_HAZARD = "log_hazard"
    MTLR = "mtlr"
    PMF = "pmf"
    # skurv models
    CGBS = "cgbs"  # Component Wise Gradient Boosting
    COX_PH_STD = "cox_ph_std"
    EST = "est"  # Extra Survival Trees
    GBS = "gbs"  # Gradient Boosting Survival
    RSF = "rsf"  # Random Survival Forest


DATA_DIR = files("ospkg") / "data"
PROJECT_ROOT = (DATA_DIR / "../..").resolve()
RESULTS_DIR = PROJECT_ROOT / "results"

MAX_OUTER_SPLITS = 5
MAX_INNER_SPLITS = 5
