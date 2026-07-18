from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

TRAIN_DATA = DATA_DIR / "KDDTrain+.txt"
TEST_DATA = DATA_DIR / "KDDTest+.txt"
FEATURE_NAMES = DATA_DIR / "feature_names.txt"

# Automatically create output folders
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

RF_ESTIMATORS = 500
LR_MAX_ITER = 5000

RULE_PERCENTILE = 0.95
RULE_MIN_SCORE = 2

HYBRID_WEIGHTS = {
    "rf": 0.6,
    "lr": 0.3,
    "rule": 0.1,
}

HYBRID_THRESHOLD = 0.5

TOP_FEATURES = 15

SHAP_SAMPLE_SIZE = 500