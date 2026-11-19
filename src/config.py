
from pathlib import Path
import numpy as np

# Project directories
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"

OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

# Dataset paths
TRAIN_DATA = DATA_DIR / "KDDTrain+.txt"
TEST_DATA = DATA_DIR / "KDDTest+.txt"

# Automatically create output directories
for directory in [OUTPUT_DIR, PLOT_DIR, MODEL_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

# Validation configuration
VALIDATION_SIZE = 0.20
CV_FOLDS = 3

# Random Forest
RF_ESTIMATORS = 500

# Logistic Regression
LR_MAX_ITER = 5000

# SHAP explainability
SHAP_SAMPLE_SIZE = 300

# Ensemble weight optimization
WEIGHT_GRID = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

# Classification threshold optimization
THRESHOLD_GRID = np.arange(0.10, 0.91, 0.02)
