
"""
Model construction for the Network Intrusion Detection System.

Creates Random Forest, Logistic Regression, and XGBoost
pipelines with integrated preprocessing.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    RF_ESTIMATORS,
    LR_MAX_ITER,
)

from src.data.preprocessing import build_preprocessor


def build_random_forest(X_train):
    """Build a Random Forest classification pipeline."""

    preprocessor = build_preprocessor(X_train)

    classifier = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_logistic_regression(X_train):
    """Build a Logistic Regression classification pipeline."""

    preprocessor = build_preprocessor(X_train)

    classifier = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )



def build_xgboost(X_train, y_train):
    """Build an XGBoost classification pipeline."""

    preprocessor = build_preprocessor(X_train)

    y_train = np.asarray(y_train)

    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)

    if positive_count == 0:
        raise ValueError("Training data contains no attack samples.")

    scale_pos_weight = negative_count / positive_count

    classifier = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
