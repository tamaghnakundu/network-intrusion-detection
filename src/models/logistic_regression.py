"""
Logistic Regression model for Hybrid Intrusion Detection System
"""

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import (
    LR_MAX_ITER,
    RANDOM_STATE,
    MODEL_DIR,
)


def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    save_model=True,
):
    """
    Train the Logistic Regression classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.

    X_test : pandas.DataFrame
        Test features.

    y_train : pandas.Series
        Training labels.

    save_model : bool, default=True
        Save the trained model and scaler.

    Returns
    -------
    model_lr : LogisticRegression
        Trained Logistic Regression model.

    scaler : StandardScaler
        Fitted scaler.

    X_train_scaled : numpy.ndarray
        Scaled training features.

    X_test_scaled : numpy.ndarray
        Scaled test features.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    model_lr = LogisticRegression(
        class_weight="balanced",
        max_iter=LR_MAX_ITER,
        random_state=RANDOM_STATE,
    )

    model_lr.fit(
        X_train_scaled,
        y_train,
    )

    if save_model:

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model_lr,
            MODEL_DIR / "logistic_regression.pkl",
        )

        joblib.dump(
            scaler,
            MODEL_DIR / "scaler.pkl",
        )

    return (
        model_lr,
        scaler,
        X_train_scaled,
        X_test_scaled,
    )


def load_logistic_regression():
    """
    Load the saved Logistic Regression model and scaler.
    """

    model_path = MODEL_DIR / "logistic_regression.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Logistic Regression model not found. Train the model first."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            "Scaler not found. Train the model first."
        )

    model_lr = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model_lr, scaler


def predict(
    model_lr,
    X_test_scaled,
):
    """
    Predict class labels.
    """

    return model_lr.predict(X_test_scaled)


def predict_proba(
    model_lr,
    X_test_scaled,
):
    """
    Predict class probabilities.
    """

    return model_lr.predict_proba(X_test_scaled)