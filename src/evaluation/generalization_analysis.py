
"""
Generalisation analysis for NSL-KDD.

This module compares detection performance on:
1. Known attack types: attack names present in KDDTrain+.
2. Unseen attack types: attack names absent from KDDTrain+.
3. Individual attack types.
4. Known/unseen attacks within each broad attack category.

The analysis is descriptive only. It does not train models, tune thresholds,
or modify predictions.

Expected metadata keys:
    metadata["train_attack_labels"]
    metadata["test_attack_labels"]

Expected predictions_by_model:
    {
        "Random Forest": rf_predictions,
        "Logistic Regression": lr_predictions,
        "XGBoost": xgb_predictions,
        "RF + LR": rf_lr_predictions,
        "RF + XGBoost": rf_xgb_predictions,
        "LR + XGBoost": lr_xgb_predictions,
        "Hybrid IDS": hybrid_predictions,
    }

Predictions and y_true must use binary encoding:
    normal = 0
    attack = 1
"""

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VALID_ATTACK_CATEGORIES = {"DOS", "PROBE", "R2L", "U2R"}


def _normalise_attack_name(value) -> str:
    """Normalise attack labels for reliable train/test membership checks."""
    return str(value).strip().lower()


def _normalise_category(value) -> str:
    """Normalise category labels while preserving consistent formatting."""
    return str(value).strip().upper()


def _validate_inputs(
    y_true: Sequence,
    train_attack_labels: Sequence,
    test_attack_labels: Sequence,
    predictions_by_model: Mapping[str, Sequence],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Validate labels, attack names and model predictions."""

    y_true = np.asarray(y_true).reshape(-1)

    train_labels = np.asarray(
        train_attack_labels, dtype=object
    ).reshape(-1)

    test_labels = np.asarray(
        test_attack_labels, dtype=object
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError("The test arrays are empty.")

    if len(train_labels) == 0:
        raise ValueError("Training attack labels are empty.")

    if len(y_true) != len(test_labels):
        raise ValueError(
            "y_true and test_attack_labels must have the same "
            "number of samples."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "y_true must use binary labels: normal=0 and attack=1."
        )

    y_true = y_true.astype(int)

    # Validate attack names before performing membership checks.
    for name, labels in (
        ("Training", train_labels),
        ("Test", test_labels),
    ):
        normalised = [
            _normalise_attack_name(value)
            for value in labels
        ]

        if any(
            not value or value in {"none", "nan"}
            for value in normalised
        ):
            raise ValueError(
                f"{name} attack labels contain missing or empty values."
            )

    # Verify that binary labels agree with original attack names.
    test_is_normal = np.array([
        _normalise_attack_name(value) == "normal"
        for value in test_labels
    ])

    expected_labels = (~test_is_normal).astype(int)

    if not np.array_equal(y_true, expected_labels):
        raise ValueError(
            "Binary labels do not match test attack names. "
            "Expected y_true=0 for normal and y_true=1 for attacks. "
            "Check metadata alignment."
        )

    predictions = {}

    for model_name, values in predictions_by_model.items():

        pred = np.asarray(values).reshape(-1)

        if len(pred) != len(y_true):
            raise ValueError(
                f"Predictions for {model_name!r} have {len(pred)} rows; "
                f"expected {len(y_true)}."
            )

        if not np.isin(pred, [0, 1]).all():
            raise ValueError(
                f"Predictions for {model_name!r} must be binary (0 or 1)."
            )

        predictions[str(model_name)] = pred.astype(int)

    if not predictions:
        raise ValueError(
            "predictions_by_model must contain at least one model."
        )

    return (
        y_true,
        train_labels,
        test_labels,
        predictions,
    )


def _validate_categories(
    y_true: np.ndarray,
    attack_categories: Sequence,
) -> np.ndarray:
    """Validate broad attack categories and their alignment with labels."""

    categories = np.asarray(
        attack_categories, dtype=object
    ).reshape(-1)

    if len(categories) != len(y_true):
        raise ValueError(
            "attack_categories and y_true must have the same "
            "number of samples."
        )

    normalised_categories = np.array([
        _normalise_category(value)
        for value in categories
    ], dtype=object)

    if any(
        not value or value in {"NONE", "NAN"}
        for value in normalised_categories
    ):
        raise ValueError(
            "Attack categories contain missing or empty values."
        )

    is_attack = y_true == 1

    invalid_attack_categories = (
        set(normalised_categories[is_attack])
        - VALID_ATTACK_CATEGORIES
    )

    if invalid_attack_categories:
        raise ValueError(
            "Invalid attack categories found: "
            f"{sorted(invalid_attack_categories)}. "
            f"Expected one of {sorted(VALID_ATTACK_CATEGORIES)}."
        )

    if not np.all(
        normalised_categories[~is_attack] == "NORMAL"
    ):
        raise ValueError(
            "Normal samples must have the category 'Normal'."
        )

    return normalised_categories


def _attack_detection_row(
    model_name: str,
    group_name: str,
    attack_mask: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Calculate detection rate and FNR for an attack-only subset."""

    support = int(attack_mask.sum())

    detected = int(
        ((y_pred == 1) & attack_mask).sum()
    )

    missed = support - detected

    return {
        "Model": model_name,
        "Attack_Group": group_name,
        "Attack_Support": support,
        "Detected": detected,
        "Missed": missed,
        "Detection_Rate": (
            detected / support if support else np.nan
        ),
        "False_Negative_Rate": (
            missed / support if support else np.nan
        ),
    }


def analyze_generalization(
    y_true: Sequence,
    train_attack_labels: Sequence,
    test_attack_labels: Sequence,
    attack_categories: Sequence,
    predictions_by_model: Mapping[str, Sequence],
    output_dir: str = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate known/unseen, per-attack, and category-level
    generalisation reports.

    Parameters
    ----------
    y_true:
        Binary KDDTest+ labels (normal=0, attack=1).

    train_attack_labels:
        Original attack-name labels from KDDTrain+.

    test_attack_labels:
        Original attack-name labels from KDDTest+.

    attack_categories:
        Broad category for each KDDTest+ row:
        Normal, DoS, Probe, R2L or U2R.

    predictions_by_model:
        Mapping of model names to binary predictions aligned with y_true.

    output_dir:
        Project output directory. CSV files are written to output_dir/reports
        and plots to output_dir/plots.

    Returns
    -------
    known_unseen_df, individual_attack_df, category_generalization_df
    """

    (
        y_true,
        train_labels,
        test_labels,
        predictions,
    ) = _validate_inputs(
        y_true,
        train_attack_labels,
        test_attack_labels,
        predictions_by_model,
    )

    categories = _validate_categories(
        y_true,
        attack_categories,
    )

    # Determine attack familiarity using original attack names.
    train_attack_names = {
        _normalise_attack_name(name)
        for name in train_labels
        if _normalise_attack_name(name) != "normal"
    }

    test_attack_names = np.array([
        _normalise_attack_name(name)
        for name in test_labels
    ], dtype=object)

    is_attack = y_true == 1

    known_mask = (
        is_attack
        & np.isin(test_attack_names, list(train_attack_names))
    )

    unseen_mask = (
        is_attack
        & ~np.isin(test_attack_names, list(train_attack_names))
    )

    # Every attack sample must belong to exactly one group.
    if not np.array_equal(
        known_mask | unseen_mask,
        is_attack,
    ):
        raise RuntimeError(
            "Known/unseen masks do not cover all attack samples."
        )

    if np.any(known_mask & unseen_mask):
        raise RuntimeError(
            "Known and unseen attack groups overlap."
        )

    known_rows = []
    individual_rows = []
    category_rows = []

    # Known-versus-unseen: attack-only detection rates.
    # These subsets contain no normal samples, so specificity,
    # FPR, MCC, ROC-AUC and AP are not appropriate metrics here.
    for model_name, y_pred in predictions.items():

        known_rows.append(
            _attack_detection_row(
                model_name,
                "Known",
                known_mask,
                y_pred,
            )
        )

        known_rows.append(
            _attack_detection_row(
                model_name,
                "Unseen",
                unseen_mask,
                y_pred,
            )
        )

        # Individual attack names, excluding normal traffic.
        unique_test_attacks = sorted(
            set(test_attack_names[is_attack])
        )

        for attack_name in unique_test_attacks:

            attack_mask = (
                is_attack
                & (test_attack_names == attack_name)
            )

            row = _attack_detection_row(
                model_name,
                attack_name,
                attack_mask,
                y_pred,
            )

            row["Seen_in_Training"] = (
                attack_name in train_attack_names
            )

            individual_rows.append(row)

        # Category x familiarity breakdown.
        category_names = sorted(
            set(categories[is_attack])
        )

        for category_name in category_names:

            category_mask = (
                is_attack
                & (categories == category_name)
            )

            for familiarity, familiarity_mask in (
                ("Known", known_mask),
                ("Unseen", unseen_mask),
            ):

                combined_mask = (
                    category_mask & familiarity_mask
                )

                category_rows.append(
                    _attack_detection_row(
                        model_name,
                        f"{category_name} - {familiarity}",
                        combined_mask,
                        y_pred,
                    )
                )

    known_unseen_df = pd.DataFrame(known_rows)
    individual_attack_df = pd.DataFrame(individual_rows)
    category_generalization_df = pd.DataFrame(category_rows)

    # Create output directories.
    output_path = Path(output_dir)

    reports_dir = output_path / "reports"
    plots_dir = output_path / "plots"

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV reports.
    known_unseen_df.to_csv(
        reports_dir / "known_vs_unseen_performance.csv",
        index=False,
    )

    individual_attack_df.to_csv(
        reports_dir / "individual_attack_performance.csv",
        index=False,
    )

    category_generalization_df.to_csv(
        reports_dir / "category_generalization_performance.csv",
        index=False,
    )

    # Generate plots.
    _plot_known_vs_unseen(
        known_unseen_df,
        plots_dir,
    )

    _plot_individual_attack_detection(
        individual_attack_df,
        plots_dir,
    )

    # Print summary.
    print("\n" + "=" * 65)
    print("GENERALISATION ANALYSIS")
    print("=" * 65)

    print(
        f"\nKnown attack samples:  {int(known_mask.sum()):,}"
    )

    print(
        f"Unseen attack samples: {int(unseen_mask.sum()):,}"
    )

    print(
        "Unique unseen attack types: "
        f"{len(set(test_attack_names[unseen_mask]))}"
    )

    print(
        "Unique known attack types: "
        f"{len(set(test_attack_names[known_mask]))}"
    )

    print("\nKnown vs Unseen Detection Performance")
    print("-" * 65)

    print(
        known_unseen_df.to_string(
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

    unseen_names = sorted(
        set(test_attack_names[unseen_mask])
    )

    if unseen_names:
        print("\nUnseen attack types:")
        print(", ".join(unseen_names))

    print("\nReports saved to:", reports_dir.resolve())
    print("Plots saved to:  ", plots_dir.resolve())

    return (
        known_unseen_df,
        individual_attack_df,
        category_generalization_df,
    )


def _plot_known_vs_unseen(
    df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Plot attack detection rates for known and unseen attack samples."""

    pivot = df.pivot(
        index="Model",
        columns="Attack_Group",
        values="Detection_Rate",
    )

    ax = pivot.reindex(
        columns=["Known", "Unseen"]
    ).plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1),
        width=0.8,
    )

    ax.set_title(
        "Detection Rate: Known vs Unseen Attack Types"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Attack Detection Rate")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
    )

    ax.legend(title="Attack Group")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        plots_dir / "known_vs_unseen_detection.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def _plot_individual_attack_detection(
    df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Plot per-attack detection rate, with one series per model."""

    if df.empty:
        return

    pivot = df.pivot(
        index="Attack_Group",
        columns="Model",
        values="Detection_Rate",
    ).sort_index()

    width = max(
        12,
        min(22, 0.45 * len(pivot.index) + 6),
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(width, 6),
        ylim=(0, 1),
        width=0.8,
    )

    ax.set_title(
        "Detection Rate by Individual Attack Type"
    )

    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Attack Detection Rate")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=65,
        ha="right",
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        plots_dir / "individual_attack_detection.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


if __name__ == "__main__":
    raise SystemExit(
        "Import analyze_generalization() from main.py and pass the "
        "aligned labels, metadata, and predictions."
    )
