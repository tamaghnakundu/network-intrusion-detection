
"""
Generalised error analysis utilities for the Network Intrusion
Detection System.

Supports any number of individual models and ensembles.

Class definitions:
    0 = Normal
    1 = Attack
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.config import REPORT_DIR


def create_error_dataframe(
    X_test,
    y_test,
    predictions_by_model,
    probabilities_by_model=None,
):
    """
    Create a dataframe containing actual labels, predictions,
    probabilities and model-wise error classifications.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Original test features.

    y_test : array-like
        Actual binary labels.

    predictions_by_model : dict
        Dictionary mapping model names to predictions.

    probabilities_by_model : dict, optional
        Dictionary mapping model names to attack probabilities.

    Returns
    -------
    pandas.DataFrame
        Combined error analysis dataframe.
    """

    y_test = np.asarray(y_test)

    n_samples = len(y_test)

    if len(X_test) != n_samples:
        raise ValueError(
            "X_test and y_test must have equal lengths."
        )

    if not np.isin(y_test, [0, 1]).all():
        raise ValueError(
            "y_test must contain only binary labels 0 and 1."
        )

    if not predictions_by_model:
        raise ValueError(
            "At least one model prediction is required."
        )

    probabilities_by_model = (
        probabilities_by_model or {}
    )

    unknown_probability_models = (
        set(probabilities_by_model)
        - set(predictions_by_model)
    )

    if unknown_probability_models:
        raise ValueError(
            "Probabilities supplied for unknown models: "
            f"{unknown_probability_models}"
        )

    error_df = X_test.copy()

    error_df["Actual"] = y_test

    for model_name, predictions in predictions_by_model.items():

        predictions = np.asarray(predictions)

        if len(predictions) != n_samples:
            raise ValueError(
                f"{model_name} predictions have an invalid length."
            )

        if not np.isin(predictions, [0, 1]).all():
            raise ValueError(
                f"{model_name} predictions must contain "
                "only binary labels 0 and 1."
            )

        error_df[f"{model_name}_Prediction"] = predictions

        error_df[f"{model_name}_Correct"] = (
            y_test == predictions
        )

        error_df[f"{model_name}_Error"] = (
            y_test != predictions
        ).astype(int)

        error_df[f"{model_name}_Error_Type"] = np.select(
            [
                (y_test == 0) & (predictions == 1),
                (y_test == 1) & (predictions == 0),
            ],
            [
                "False Positive",
                "False Negative",
            ],
            default="Correct",
        )

        if model_name in probabilities_by_model:

            probabilities = np.asarray(
                probabilities_by_model[model_name],
                dtype=float,
            )

            if len(probabilities) != n_samples:
                raise ValueError(
                    f"{model_name} probabilities have "
                    "an invalid length."
                )

            if not np.isfinite(probabilities).all():
                raise ValueError(
                    f"{model_name} probabilities contain "
                    "non-finite values."
                )

            if (
                (probabilities < 0).any()
                or (probabilities > 1).any()
            ):
                raise ValueError(
                    f"{model_name} probabilities must be "
                    "between 0 and 1."
                )

            error_df[
                f"{model_name}_Attack_Probability"
            ] = probabilities

    return error_df


def get_misclassified_samples(
    error_df,
    model_name,
):
    """Return samples misclassified by a specified model."""

    column = f"{model_name}_Error"

    if column not in error_df.columns:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return error_df[
        error_df[column] == 1
    ].copy()


def split_errors(
    error_df,
    model_name,
):
    """Separate false positives and false negatives."""

    column = f"{model_name}_Error_Type"

    if column not in error_df.columns:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    fp = error_df[
        error_df[column] == "False Positive"
    ].copy()

    fn = error_df[
        error_df[column] == "False Negative"
    ].copy()

    return fp, fn


def calculate_error_summary(error_df):
    """
    Calculate model-wise error counts, false positives,
    false negatives and error rates.
    """

    summary = []

    total_samples = len(error_df)

    model_names = [
        column[:-len("_Prediction")]
        for column in error_df.columns
        if column.endswith("_Prediction")
    ]

    for model_name in model_names:

        error_column = f"{model_name}_Error"
        error_type_column = f"{model_name}_Error_Type"

        if (
            error_column not in error_df.columns
            or error_type_column not in error_df.columns
        ):
            raise ValueError(
                f"Incomplete error information for {model_name}."
            )

        incorrect = error_df[error_column] == 1

        fp_count = int(
            (
                error_df[error_type_column]
                == "False Positive"
            ).sum()
        )

        fn_count = int(
            (
                error_df[error_type_column]
                == "False Negative"
            ).sum()
        )

        error_count = int(incorrect.sum())

        summary.append({
            "Model": model_name,
            "Total_Samples": total_samples,
            "Correct_Predictions": total_samples - error_count,
            "Incorrect_Predictions": error_count,
            "False_Positives": fp_count,
            "False_Negatives": fn_count,
            "Error_Rate": (
                error_count / total_samples
                if total_samples > 0
                else 0.0
            ),
        })

    return pd.DataFrame(summary)


def _safe_filename(model_name):
    """Convert model names into filesystem-safe filenames."""

    filename = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        model_name.lower(),
    )

    return filename.strip("_")


def save_error_analysis(
    error_df,
    output_dir=REPORT_DIR,
):
    """
    Save model-wise error analysis for every evaluated model.

    Outputs
    -------
    error_analysis.csv
    error_summary.csv
    {model}_misclassified.csv
    {model}_false_positives.csv
    {model}_false_negatives.csv
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_df.to_csv(
        output_dir / "error_analysis.csv",
        index=False,
    )

    summary = calculate_error_summary(error_df)

    summary.to_csv(
        output_dir / "error_summary.csv",
        index=False,
    )

    for model_name in summary["Model"].tolist():

        misclassified = get_misclassified_samples(
            error_df,
            model_name,
        )

        fp, fn = split_errors(
            error_df,
            model_name,
        )

        safe_name = _safe_filename(model_name)

        misclassified.to_csv(
            output_dir / f"{safe_name}_misclassified.csv",
            index=False,
        )

        fp.to_csv(
            output_dir / f"{safe_name}_false_positives.csv",
            index=False,
        )

        fn.to_csv(
            output_dir / f"{safe_name}_false_negatives.csv",
            index=False,
        )

    print("\nModel-wise Error Analysis")
    print("-" * 65)
    print(summary.to_string(index=False))

    print(
        f"\nError analysis saved to: {output_dir}"
    )
