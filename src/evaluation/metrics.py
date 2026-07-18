"""
Evaluation utilities for the Hybrid Intrusion Detection System.
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import REPORT_DIR


def evaluate_model(
    model_name,
    y_true,
    y_pred,
    save_report=True,
):
    """
    Evaluate a classification model.

    Parameters
    ----------
    model_name : str
        Name of the model.

    y_true : array-like

    y_pred : array-like

    save_report : bool
        Save classification report to outputs/reports.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

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

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")
    print(report)

    if save_report:

        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            REPORT_DIR / f"{model_name.lower().replace(' ', '_')}_report.txt",
            "w",
        ) as f:

            f.write(report)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def save_predictions(
    filename,
    y_true,
    y_pred,
):
    """
    Save predictions to CSV.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame({
        "Actual": y_true,
        "Prediction": y_pred,
    })

    df.to_csv(
        REPORT_DIR / filename,
        index=False,
    )