
"""
Model training and probability calibration.

The original NSL-KDD training dataset is divided into
training and validation subsets using stratified sampling.

Random Forest, Logistic Regression, and XGBoost are trained
using CalibratedClassifierCV with sigmoid calibration.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.config import (
    RANDOM_STATE,
    VALIDATION_SIZE,
    CV_FOLDS,
)

from src.models.build_models import (
    build_random_forest,
    build_logistic_regression,
    build_xgboost,
)


def split_training_data(X_train, y_train):
    """
    Split the original training data into training and
    validation subsets.

    Stratification preserves the class distribution
    in both subsets.
    """

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    return X_fit, X_val, y_fit, y_val


def train_models(X_fit, y_fit):
    """
    Train calibrated Random Forest, Logistic Regression,
    and XGBoost models.

    Probability calibration is performed using stratified
    cross-validation on the training subset only.

    Returns
    -------
    calibrated_rf : CalibratedClassifierCV
    calibrated_lr : CalibratedClassifierCV
    calibrated_xgb : CalibratedClassifierCV
    """

    # Construct independent model pipelines.
    rf_model = build_random_forest(X_fit)
    lr_model = build_logistic_regression(X_fit)
    xgb_model = build_xgboost(X_fit, y_fit)

    # Stratified cross-validation for probability calibration.
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Calibrate Random Forest probabilities.
    calibrated_rf = CalibratedClassifierCV(
        estimator=rf_model,
        method="sigmoid",
        cv=cv,
        n_jobs=1,
    )

    # Calibrate Logistic Regression probabilities.
    calibrated_lr = CalibratedClassifierCV(
        estimator=lr_model,
        method="sigmoid",
        cv=cv,
        n_jobs=1,
    )

    # Calibrate XGBoost probabilities.
    calibrated_xgb = CalibratedClassifierCV(
        estimator=xgb_model,
        method="sigmoid",
        cv=cv,
        n_jobs=1,
    )

    # Fit models using only the training subset.
    calibrated_rf.fit(X_fit, y_fit)
    calibrated_lr.fit(X_fit, y_fit)
    calibrated_xgb.fit(X_fit, y_fit)

    return calibrated_rf, calibrated_lr, calibrated_xgb
