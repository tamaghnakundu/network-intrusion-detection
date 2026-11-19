
"""
Attack-category analysis for the Network Intrusion Detection System.

The classifiers perform binary classification:
    0 = Normal
    1 = Attack

This module evaluates how effectively each model detects samples
belonging to individual attack families. It does not perform
multiclass attack-family classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import REPORT_DIR, PLOT_DIR


ATTACK_CATEGORIES = [
    "DoS",
    "Probe",
    "R2L",
    "U2R",
]


def _validate_inputs(y_true, attack_categories, predictions_by_model):
    """Validate labels, categories and model predictions."""

    y_true = np.asarray(y_true)
    attack_categories = np.asarray(attack_categories).astype(str)

    if y_true.ndim != 1 or attack_categories.ndim != 1:
        raise ValueError("Labels and attack categories must be one-dimensional.")

    if len(y_true) != len(attack_categories):
        raise ValueError(
            "y_true and attack_categories must have equal lengths."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1.")

    y_true = y_true.astype(int)

    allowed_categories = {"Normal", *ATTACK_CATEGORIES}
    unique_categories = set(np.unique(attack_categories))

    if not unique_categories.issubset(allowed_categories):
        unknown = sorted(unique_categories - allowed_categories)
        raise ValueError(f"Unknown attack categories found: {unknown}")

    if not np.array_equal(
        y_true,
        (attack_categories != "Normal").astype(int),
    ):
        raise ValueError(
            "Binary labels do not match the supplied attack categories."
        )

    validated_predictions = {}

    for model_name, predictions in predictions_by_model.items():

        predictions = np.asarray(predictions)

        if predictions.ndim != 1:
            raise ValueError(
                f"{model_name}: predictions must be one-dimensional."
            )

        if len(predictions) != len(y_true):
            raise ValueError(
                f"{model_name}: prediction length does not match y_true."
            )

        if not np.isin(predictions, [0, 1]).all():
            raise ValueError(
                f"{model_name}: predictions must contain only 0 and 1."
            )

        validated_predictions[model_name] = predictions.astype(int)

    if not validated_predictions:
        raise ValueError("predictions_by_model cannot be empty.")

    return y_true, attack_categories, validated_predictions


def calculate_attack_category_metrics(
    y_true,
    attack_categories,
    predictions_by_model,
):
    """
    Calculate attack detection metrics for each attack category.

    For each attack family:

        Detection Rate = Detected attacks / Total attacks

        False Negative Rate = Missed attacks / Total attacks

    These are category-specific recall and miss-rate measurements.
    They do not represent multiclass classification metrics.
    """

    (
        y_true,
        attack_categories,
        predictions_by_model,
    ) = _validate_inputs(
        y_true,
        attack_categories,
        predictions_by_model,
    )

    results = []

    # Iterate over every supplied model, including XGBoost and ensembles.
    for model_name, predictions in predictions_by_model.items():

        for category in ATTACK_CATEGORIES:

            category_mask = attack_categories == category
            total_samples = int(category_mask.sum())

            if total_samples == 0:
                detected = 0
                missed = 0
                detection_rate = np.nan
                false_negative_rate = np.nan

            else:
                category_predictions = predictions[category_mask]

                detected = int(
                    np.sum(category_predictions == 1)
                )

                missed = int(
                    np.sum(category_predictions == 0)
                )

                detection_rate = detected / total_samples
                false_negative_rate = missed / total_samples

            results.append({
                "Model": model_name,
                "Attack_Category": category,
                "Total_Samples": total_samples,
                "Detected_Attacks": detected,
                "Missed_Attacks": missed,
                "Detection_Rate": detection_rate,
                "False_Negative_Rate": false_negative_rate,
            })

    return pd.DataFrame(results)


def calculate_normal_traffic_metrics(
    attack_categories,
    predictions_by_model,
):
    """
    Calculate false alarms and normal-traffic rejection rates.

    A false alarm occurs when a normal sample is predicted as attack.
    """

    attack_categories = np.asarray(attack_categories).astype(str)

    if attack_categories.ndim != 1:
        raise ValueError("attack_categories must be one-dimensional.")

    normal_mask = attack_categories == "Normal"
    total_normal = int(normal_mask.sum())

    results = []

    for model_name, model_predictions in predictions_by_model.items():

        predictions = np.asarray(model_predictions)

        if predictions.ndim != 1:
            raise ValueError(
                f"{model_name}: predictions must be one-dimensional."
            )

        if len(predictions) != len(attack_categories):
            raise ValueError(
                f"{model_name}: prediction length does not match "
                "attack_categories."
            )

        if not np.isin(predictions, [0, 1]).all():
            raise ValueError(
                f"{model_name}: predictions must contain only 0 and 1."
            )

        predictions = predictions.astype(int)
        normal_predictions = predictions[normal_mask]

        false_alarms = int(
            np.sum(normal_predictions == 1)
        )

        correctly_rejected = int(
            np.sum(normal_predictions == 0)
        )

        if total_normal > 0:
            false_alarm_rate = false_alarms / total_normal
            normal_rejection_rate = (
                correctly_rejected / total_normal
            )
        else:
            false_alarm_rate = np.nan
            normal_rejection_rate = np.nan

        results.append({
            "Model": model_name,
            "Total_Normal_Samples": total_normal,
            "Correctly_Classified_Normal": correctly_rejected,
            "False_Alarms": false_alarms,
            "False_Alarm_Rate": false_alarm_rate,
            "Normal_Rejection_Rate": normal_rejection_rate,
        })

    return pd.DataFrame(results)


def plot_attack_detection_rates(
    category_results,
    output_dir,
):
    """Generate a grouped bar chart of category detection rates."""

    pivot = category_results.pivot(
        index="Attack_Category",
        columns="Model",
        values="Detection_Rate",
    )

    pivot = pivot.reindex(ATTACK_CATEGORIES)

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
    )

    ax.set_title("Attack Detection Rate by Category")
    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Detection Rate")
    ax.set_ylim(0, 1.05)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = (
        output_dir / "attack_category_detection_rates.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def plot_attack_false_negatives(
    category_results,
    output_dir,
):
    """Generate a grouped bar chart of missed attacks per category."""

    pivot = category_results.pivot(
        index="Attack_Category",
        columns="Model",
        values="Missed_Attacks",
    )

    pivot = pivot.reindex(ATTACK_CATEGORIES)

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
        width=0.8,
    )

    ax.set_title("False Negatives by Attack Category")
    ax.set_xlabel("Attack Category")
    ax.set_ylabel("Number of Missed Attacks")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=0,
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = (
        output_dir / "attack_category_false_negatives.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def analyze_attack_categories(
    y_true,
    attack_categories,
    predictions_by_model,
    report_dir=REPORT_DIR,
    plot_dir=PLOT_DIR,
):
    """
    Run the complete attack-category analysis.

    Parameters
    ----------
    y_true : array-like
        Binary test labels: 0 = Normal, 1 = Attack.

    attack_categories : array-like
        Original test attack categories:
        Normal, DoS, Probe, R2L or U2R.

    predictions_by_model : dict
        Model predictions, for example:
        {
            "Random Forest": rf_predictions,
            "Logistic Regression": lr_predictions,
            "XGBoost": xgb_predictions,
            "RF + LR": rf_lr_predictions,
            "RF + XGBoost": rf_xgb_predictions,
            "LR + XGBoost": lr_xgb_predictions,
            "Hybrid IDS": hybrid_predictions
        }

    report_dir : str or Path
        Directory for CSV reports.

    plot_dir : str or Path
        Directory for plots.

    Returns
    -------
    tuple
        Attack-category results and normal-traffic results.
    """

    report_dir = Path(report_dir)
    plot_dir = Path(plot_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    (
        y_true,
        attack_categories,
        predictions_by_model,
    ) = _validate_inputs(
        y_true,
        attack_categories,
        predictions_by_model,
    )

    category_results = calculate_attack_category_metrics(
        y_true,
        attack_categories,
        predictions_by_model,
    )

    normal_results = calculate_normal_traffic_metrics(
        attack_categories,
        predictions_by_model,
    )

    category_csv = (
        report_dir / "attack_category_performance.csv"
    )

    normal_csv = (
        report_dir / "normal_traffic_analysis.csv"
    )

    category_results.to_csv(
        category_csv,
        index=False,
    )

    normal_results.to_csv(
        normal_csv,
        index=False,
    )

    detection_plot = plot_attack_detection_rates(
        category_results,
        plot_dir,
    )

    false_negative_plot = plot_attack_false_negatives(
        category_results,
        plot_dir,
    )

    print("\n" + "=" * 65)
    print("ATTACK-CATEGORY ANALYSIS")
    print("=" * 65)

    print("\nCategory-wise Detection Performance")
    print("-" * 65)

    print(
        category_results.to_string(
            index=False,
            formatters={
                "Detection_Rate": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "N/A"
                ),
                "False_Negative_Rate": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    print("\nNormal Traffic Analysis")
    print("-" * 65)

    print(
        normal_results.to_string(
            index=False,
            formatters={
                "False_Alarm_Rate": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "N/A"
                ),
                "Normal_Rejection_Rate": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    print("\nSaved Reports")
    print("-" * 65)
    print(f"Category performance : {category_csv}")
    print(f"Normal traffic       : {normal_csv}")

    print("\nSaved Plots")
    print("-" * 65)
    print(f"Detection rates      : {detection_plot}")
    print(f"False negatives      : {false_negative_plot}")

    return category_results, normal_results
