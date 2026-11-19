
"""
Ensemble learning for the Network Intrusion Detection System.

Supports weighted probability fusion of:
    - Random Forest
    - Logistic Regression
    - XGBoost

Ensemble weights and classification thresholds are
optimized using validation data only.
"""

import itertools
import numpy as np

from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    recall_score,
)

from src.config import (
    WEIGHT_GRID,
    THRESHOLD_GRID,
)


def get_attack_probabilities(model, X):
    """
    Extract predicted probabilities for attack class (1).
    """

    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)

    attack_indices = np.where(classes == 1)[0]

    if len(attack_indices) == 0:
        raise ValueError(
            "The model has no probability column for attack class 1."
        )

    return probabilities[:, attack_indices[0]]


def get_model_probabilities(models, X):
    """
    Obtain attack probabilities from multiple calibrated models.

    Parameters
    ----------
    models : dict
        Dictionary of model names and calibrated classifiers.

    X : pandas.DataFrame
        Input features.

    Returns
    -------
    dict
        Attack probabilities for each model.
    """

    if not models:
        raise ValueError("At least one model must be provided.")

    probabilities = {}

    for name, model in models.items():
        probabilities[name] = get_attack_probabilities(model, X)

    lengths = {
        len(values) for values in probabilities.values()
    }

    if len(lengths) != 1:
        raise ValueError(
            "All models must produce the same number of predictions."
        )

    return probabilities


def fuse_probabilities(probabilities_by_model, weights):
    """
    Combine model probabilities using weighted probability fusion.

    Parameters
    ----------
    probabilities_by_model : dict
        Dictionary containing model probabilities.

    weights : dict
        Weight assigned to each model.

    Returns
    -------
    numpy.ndarray
        Fused attack probabilities.
    """

    if set(probabilities_by_model) != set(weights):
        raise ValueError(
            "Probability and weight dictionaries must contain "
            "the same model names."
        )

    weight_values = np.asarray(
        list(weights.values()),
        dtype=float,
    )

    if np.any(weight_values < 0):
        raise ValueError("Weights cannot be negative.")

    if not np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Ensemble weights must sum to 1.")

    fused_probabilities = np.zeros(
        len(next(iter(probabilities_by_model.values()))),
        dtype=float,
    )

    for name, probabilities in probabilities_by_model.items():
        fused_probabilities += (
            weights[name] * probabilities
        )

    return fused_probabilities


def generate_weight_combinations(model_names):
    """
    Generate valid weight combinations from WEIGHT_GRID.

    All weights are non-negative and sum to 1.

    The final model's weight is calculated as the remaining
    weight, avoiding unnecessary combinations.
    """

    model_names = list(model_names)

    number_of_models = len(model_names)

    weight_grid = sorted(
        set(float(weight) for weight in WEIGHT_GRID)
    )

    if number_of_models == 1:
        yield {model_names[0]: 1.0}
        return

    if not weight_grid:
        raise ValueError("WEIGHT_GRID cannot be empty.")

    if not np.isclose(weight_grid[0], 0.0):
        raise ValueError("WEIGHT_GRID must include 0.0.")

    if not np.isclose(weight_grid[-1], 1.0):
        raise ValueError("WEIGHT_GRID must include 1.0.")

    tolerance = 1e-9

    for partial_weights in itertools.product(
        weight_grid,
        repeat=number_of_models - 1,
    ):

        remaining_weight = 1.0 - sum(partial_weights)

        if remaining_weight < -tolerance:
            continue

        if not any(
            np.isclose(remaining_weight, weight, atol=tolerance)
            for weight in weight_grid
        ):
            continue

        weights = {
            name: float(weight)
            for name, weight in zip(
                model_names[:-1],
                partial_weights,
            )
        }

        weights[model_names[-1]] = float(
            max(0.0, remaining_weight)
        )

        yield weights


def optimize_ensemble(
    models,
    X_val,
    y_val,
):
    """
    Optimize model weights and classification threshold
    using validation data.

    Optimization priority:
        1. Highest MCC
        2. Highest F1-score
        3. Highest attack recall

    Parameters
    ----------
    models : dict
        Dictionary of calibrated models.

        Example:
        {
            "rf": calibrated_rf,
            "lr": calibrated_lr,
            "xgb": calibrated_xgb
        }

    X_val : pandas.DataFrame
        Validation features.

    y_val : pandas.Series
        Validation labels.

    Returns
    -------
    dict
        Optimal weights, threshold and validation metrics.
    """

    probabilities_by_model = get_model_probabilities(
        models,
        X_val,
    )

    y_true = np.asarray(y_val)

    best_score = None
    best_parameters = None

    weight_combinations = generate_weight_combinations(
        models.keys()
    )

    for weights in weight_combinations:

        fused_probabilities = fuse_probabilities(
            probabilities_by_model,
            weights,
        )

        for threshold in THRESHOLD_GRID:

            predictions = (
                fused_probabilities >= threshold
            ).astype(int)

            mcc = matthews_corrcoef(
                y_true,
                predictions,
            )

            f1 = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                predictions,
                zero_division=0,
            )

            current_score = (
                mcc,
                f1,
                recall,
            )

            if (
                best_score is None
                or current_score > best_score
            ):

                best_score = current_score

                best_parameters = {
                    "weights": weights.copy(),
                    "threshold": float(threshold),
                    "validation_mcc": float(mcc),
                    "validation_f1": float(f1),
                    "validation_recall": float(recall),
                }

    if best_parameters is None:
        raise RuntimeError(
            "No valid ensemble configuration was evaluated."
        )

    return best_parameters


def predict_ensemble(
    models,
    X,
    ensemble_parameters,
):
    """
    Generate ensemble probabilities and binary predictions.

    Parameters
    ----------
    models : dict
        Dictionary of calibrated models.

    X : pandas.DataFrame
        Input features.

    ensemble_parameters : dict
        Parameters returned by optimize_ensemble().

    Returns
    -------
    predictions : numpy.ndarray
        Binary predictions.

    fused_probabilities : numpy.ndarray
        Fused attack probabilities.
    """

    probabilities_by_model = get_model_probabilities(
        models,
        X,
    )

    fused_probabilities = fuse_probabilities(
        probabilities_by_model,
        ensemble_parameters["weights"],
    )

    threshold = ensemble_parameters["threshold"]

    predictions = (
        fused_probabilities >= threshold
    ).astype(int)

    return predictions, fused_probabilities


def optimize_threshold(y_true, probabilities):
    """
    Find the classification threshold that optimizes
    MCC, with F1-score and recall as tie-breakers.

    Uses validation data only.
    """

    y_true = np.asarray(y_true)

    best_score = None
    best_parameters = None

    for threshold in THRESHOLD_GRID:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        mcc = matthews_corrcoef(
            y_true,
            predictions,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        current_score = (
            mcc,
            f1,
            recall,
        )

        if (
            best_score is None
            or current_score > best_score
        ):

            best_score = current_score

            best_parameters = {
                "threshold": float(threshold),
                "validation_mcc": float(mcc),
                "validation_f1": float(f1),
                "validation_recall": float(recall),
            }

    return best_parameters
