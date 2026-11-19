
"""
Evaluation utilities for the Network Intrusion Detection System.

Provides binary classification metrics, classification reports,
confusion matrices and prediction exports.

Class definitions:
    0 = Normal
    1 = Attack
"""

import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

from src.config import REPORT_DIR


def _validate_binary_labels(y_true, y_pred):
    """Validate binary labels and matching array lengths."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if len(y_true) == 0:
        raise ValueError("Evaluation arrays cannot be empty.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1.")

    return y_true.astype(int), y_pred.astype(int)


def _validate_probabilities(y_prob, expected_length):
    """Validate predicted probabilities for the positive class (Attack)."""

    try:
        y_prob = np.asarray(y_prob, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "y_prob must contain numeric probabilities."
        ) from exc

    if y_prob.ndim != 1:
        raise ValueError("y_prob must be one-dimensional.")

    if len(y_prob) != expected_length:
        raise ValueError(
            "y_prob and y_true must have equal lengths."
        )

    if not np.isfinite(y_prob).all():
        raise ValueError(
            "y_prob contains non-finite values."
        )

    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError(
            "y_prob must contain probabilities between 0 and 1."
        )

    return y_prob


def _safe_filename(model_name):
    """Convert a model name into a safe filename."""

    filename = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(model_name).strip().lower(),
    )

    filename = filename.strip("_")

    return filename or "model"


def evaluate_model(
    model_name,
    y_true,
    y_pred,
    y_prob=None,
    save_report=True,
):
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    model_name : str
        Display name of the model.

    y_true : array-like
        Actual binary labels: 0 = Normal, 1 = Attack.

    y_pred : array-like
        Predicted binary labels.

    y_prob : array-like, optional
        Predicted probabilities for class 1 (Attack).

    save_report : bool
        Whether to save the evaluation report.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    y_true, y_pred = _validate_binary_labels(
        y_true,
        y_pred,
    )

    # Classification metrics.
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # Confusion matrix.
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # Intrusion detection metrics.
    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    # Probability-based metrics.
    roc_auc = None
    average_precision = None

    if y_prob is not None:

        y_prob = _validate_probabilities(
            y_prob,
            len(y_true),
        )

        # ROC-AUC and AP require both classes to be present.
        if len(np.unique(y_true)) == 2:

            roc_auc = roc_auc_score(
                y_true,
                y_prob,
            )

            average_precision = average_precision_score(
                y_true,
                y_prob,
            )

    # Classification report.
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )

    # Collect metrics.
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Display results.
    print("\n" + "=" * 65)
    print(str(model_name).upper())
    print("=" * 65)

    for name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "mcc",
        "roc_auc",
        "average_precision",
    ]:

        value = metrics[name]

        display_name = name.replace(
            "_",
            " ",
        ).title()

        if value is None or pd.isna(value):
            print(f"{display_name:<25}: N/A")
        else:
            print(
                f"{display_name:<25}: {value:.4f}"
            )

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report\n")
    print(report)

    # Save report.
    if save_report:

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path = (
            REPORT_DIR
            / f"{_safe_filename(model_name)}_report.txt"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                f"Model: {model_name}\n\n"
            )

            file.write("Evaluation Metrics\n")
            file.write("-" * 40 + "\n")

            for name, value in metrics.items():

                if value is None or pd.isna(value):
                    file.write(f"{name}: N/A\n")

                elif isinstance(
                    value,
                    (float, np.floating),
                ):
                    file.write(
                        f"{name}: {value:.6f}\n"
                    )

                else:
                    file.write(
                        f"{name}: {value}\n"
                    )

            file.write("\nConfusion Matrix\n")
            file.write(str(cm))

            file.write(
                "\n\nClassification Report\n"
            )

            file.write(report)

    return metrics


def save_predictions(
    filename,
    y_true,
    y_pred,
    y_prob=None,
):
    """
    Save actual labels, predictions and optional attack probabilities.

    Parameters
    ----------
    filename : str
        Output CSV filename.

    y_true : array-like
        Actual binary labels.

    y_pred : array-like
        Predicted binary labels.

    y_prob : array-like, optional
        Predicted probabilities for class 1 (Attack).
    """

    y_true, y_pred = _validate_binary_labels(
        y_true,
        y_pred,
    )

    data = {
        "Actual": y_true,
        "Prediction": y_pred,
    }

    if y_prob is not None:

        y_prob = _validate_probabilities(
            y_prob,
            len(y_true),
        )

        data["Attack_Probability"] = y_prob

    df = pd.DataFrame(data)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORT_DIR / filename

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Predictions saved to: {output_path}")

    return output_path
