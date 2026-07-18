"""
Random Forest model for Hybrid Intrusion Detection System
"""

from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RF_ESTIMATORS,
    RANDOM_STATE,
    MODEL_DIR,
)


def train_random_forest(X_train, y_train, save_model=True):
    """
    Train the Random Forest classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training labels.

    save_model : bool, default=True
        Save the trained model to disk.

    Returns
    -------
    RandomForestClassifier
        Trained Random Forest model.
    """

    model_rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model_rf.fit(X_train, y_train)

    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_rf, MODEL_DIR / "random_forest.pkl")

    return model_rf


def load_random_forest():
    """
    Load a previously saved Random Forest model.
    """

    model_path = MODEL_DIR / "random_forest.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained Random Forest model not found. "
            "Train the model first."
        )

    return joblib.load(model_path)


def predict(model_rf, X_test):
    """
    Predict class labels.
    """

    return model_rf.predict(X_test)


def predict_proba(model_rf, X_test):
    """
    Predict attack probabilities.
    """

    return model_rf.predict_proba(X_test)


def get_feature_importance(model_rf, feature_names):
    """
    Return feature importance as a sorted DataFrame.
    """

    import pandas as pd

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model_rf.feature_importances_,
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance