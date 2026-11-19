
"""
Main entry point for the Hybrid Intrusion Detection System.

Pipeline:
1. Load NSL-KDD dataset.
2. Split KDDTrain+ into training and validation partitions.
3. Train calibrated Random Forest and Logistic Regression models.
4. Optimise individual thresholds using validation data.
5. Optimise hybrid fusion weights and threshold using validation data.
6. Evaluate all models on the untouched KDDTest+ dataset.
7. Save metrics, predictions, error analysis, and SHAP plots.
"""

import json
import numpy as np
import pandas as pd

from src.config import REPORT_DIR

from src.data.loader import load_nsl_kdd

from src.models.train import (
    split_training_data,
    train_models,
)

from src.models.ensemble import (
    get_attack_probabilities,
    optimize_ensemble,
    predict_ensemble,
    optimize_threshold,
)

from src.evaluation.metrics import (
    evaluate_model,
    save_predictions,
)

from src.evaluation.error_analysis import (
    create_error_dataframe,
    save_error_analysis,
)

from src.explainability.shap_analysis import (
    SHAPExplainer,
)
from src.evaluation.attack_analysis import analyze_attack_categories
from src.evaluation.generalization_analysis import analyze_generalization
def main():

    print("=" * 65)
    print("HYBRID INTRUSION DETECTION SYSTEM")
    print("=" * 65)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------

    print("\n[1/8] Loading NSL-KDD dataset...")

    (X_train,y_train,X_test,y_test,metadata,) = load_nsl_kdd(return_metadata=True)

    print(f"Training samples  : {len(X_train)}")
    print(f"Testing samples   : {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    print("\nTest Attack Category Distribution:")
    print(
        metadata["test_attack_categories"]
        .value_counts()
    )

    print("\nUnmapped Attack Labels:")

    unknown_labels = metadata["test_attack_labels"][
        metadata["test_attack_categories"] == "Unknown"
    ].unique()

    print(unknown_labels)


    # ---------------------------------------------------------
    # ATTACK DISTRIBUTION AND UNSEEN ATTACK ANALYSIS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING AND TESTING ATTACK DISTRIBUTION")
    print("=" * 70)

    train_labels = (
        metadata["train_attack_labels"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    test_labels = (
        metadata["test_attack_labels"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    train_categories = metadata["train_attack_categories"]
    test_categories = metadata["test_attack_categories"]

    print("\nTraining Category Distribution")
    print("-" * 70)
    print(train_categories.value_counts())

    print("\nTesting Category Distribution")
    print("-" * 70)
    print(test_categories.value_counts())


    # Compare individual attack types.

    train_counts = train_labels.value_counts()
    test_counts = test_labels.value_counts()

    attack_distribution = pd.DataFrame({
        "Train_Samples": train_counts,
        "Test_Samples": test_counts,
    }).fillna(0).astype(int)

    attack_distribution["Present_In_Training"] = (
        attack_distribution["Train_Samples"] > 0
    )

    attack_distribution = attack_distribution.sort_values(
        by="Test_Samples",
        ascending=False,
    )

    print("\nIndividual Attack Distribution")
    print("-" * 70)
    print(attack_distribution.to_string())


    # Identify attack types appearing in testing but not training.

    unseen_attacks = attack_distribution[
        (attack_distribution["Test_Samples"] > 0)
        & (attack_distribution["Train_Samples"] == 0)
    ]

    print("\nUnseen Test Attack Types")
    print("-" * 70)

    if unseen_attacks.empty:
        print("No unseen attack types found.")

    else:
        print(unseen_attacks.to_string())

    print("\nDistribution analysis completed.")


    # ---------------------------------------------------------
    # 2. Split Training Data
    # ---------------------------------------------------------

    print("\n[2/8] Creating training and validation partitions...")

    X_fit, X_val, y_fit, y_val = split_training_data(
        X_train,
        y_train,
    )

    print(f"Model training samples : {len(X_fit)}")
    print(f"Validation samples     : {len(X_val)}")

    # ---------------------------------------------------------
    # 3. Train Models
    # ---------------------------------------------------------

    print("\n[3/8] Training calibrated models...")

    calibrated_rf, calibrated_lr = train_models(
        X_fit,
        y_fit,
    )

    print("Random Forest training completed.")
    print("Logistic Regression training completed.")

    # ---------------------------------------------------------
    # 4. Optimise Individual Model Thresholds
    # ---------------------------------------------------------

    print("\n[4/8] Optimising individual model thresholds...")

    rf_val_probabilities = get_attack_probabilities(
        calibrated_rf,
        X_val,
    )

    lr_val_probabilities = get_attack_probabilities(
        calibrated_lr,
        X_val,
    )

    rf_parameters = optimize_threshold(
        y_val,
        rf_val_probabilities,
    )

    lr_parameters = optimize_threshold(
        y_val,
        lr_val_probabilities,
    )

    print("\nIndividual Model Parameters")
    print("-" * 45)

    print("Random Forest:", rf_parameters)
    print("Logistic Regression:", lr_parameters)

    # ---------------------------------------------------------
    # 5. Optimise Hybrid Ensemble
    # ---------------------------------------------------------

    print("\n[5/8] Optimising hybrid ensemble...")

    ensemble_parameters = optimize_ensemble(
        calibrated_rf,
        calibrated_lr,
        X_val,
        y_val,
    )

    print("\nFinal Selected Parameters")
    print("-" * 45)

    print(
        f"Random Forest threshold      : "
        f"{rf_parameters['threshold']:.3f}"
    )

    print(
        f"Logistic Regression threshold: "
        f"{lr_parameters['threshold']:.3f}"
    )

    print(
        f"Hybrid RF weight             : "
        f"{ensemble_parameters['rf_weight']:.3f}"
    )

    print(
        f"Hybrid LR weight             : "
        f"{ensemble_parameters['lr_weight']:.3f}"
    )

    print(
        f"Hybrid threshold             : "
        f"{ensemble_parameters['threshold']:.3f}"
    )

    # ---------------------------------------------------------
    # Save Optimisation Parameters
    # ---------------------------------------------------------

    parameters_path = REPORT_DIR / "ensemble_parameters.json"

    all_parameters = {
        "random_forest": rf_parameters,
        "logistic_regression": lr_parameters,
        "hybrid_ensemble": ensemble_parameters,
    }

    with open(parameters_path, "w", encoding="utf-8") as file:

        json.dump(
            all_parameters,
            file,
            indent=4,
        )

    print(f"\nOptimisation parameters saved to: {parameters_path}")

    # ---------------------------------------------------------
    # 6. Evaluate Models on Untouched Test Data
    # ---------------------------------------------------------

    print("\n[6/8] Evaluating models on KDDTest+...")

    rf_probabilities = get_attack_probabilities(
        calibrated_rf,
        X_test,
    )

    lr_probabilities = get_attack_probabilities(
        calibrated_lr,
        X_test,
    )

    # Apply validation-optimised individual thresholds.

    rf_predictions = (
        rf_probabilities >= rf_parameters["threshold"]
    ).astype(int)

    lr_predictions = (
        lr_probabilities >= lr_parameters["threshold"]
    ).astype(int)

    # Generate hybrid predictions.

    hybrid_predictions, hybrid_probabilities = predict_ensemble(
        calibrated_rf,
        calibrated_lr,
        X_test,
        ensemble_parameters,
    )

    # Evaluate each model.

    print("\nEvaluating Random Forest...")

    rf_results = evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
        rf_probabilities,
    )

    print("\nEvaluating Logistic Regression...")

    lr_results = evaluate_model(
        "Logistic Regression",
        y_test,
        lr_predictions,
        lr_probabilities,
    )

    print("\nEvaluating Hybrid IDS...")

    hybrid_results = evaluate_model(
        "Hybrid IDS",
        y_test,
        hybrid_predictions,
        hybrid_probabilities,
    )

    # ---------------------------------------------------------
    # 7. Save Predictions and Comparison
    # ---------------------------------------------------------

    print("\n[7/8] Saving predictions and evaluation results...")

    save_predictions(
        "random_forest_predictions.csv",
        y_test,
        rf_predictions,
        rf_probabilities,
    )

    save_predictions(
        "logistic_regression_predictions.csv",
        y_test,
        lr_predictions,
        lr_probabilities,
    )

    save_predictions(
        "hybrid_predictions.csv",
        y_test,
        hybrid_predictions,
        hybrid_probabilities,
    )

    # Create consolidated model comparison.
    # Average Precision is included automatically through
    # the average_precision key returned by evaluate_model().

    comparison_metrics = [
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
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    comparison = pd.DataFrame([
        {
            "Model": "Random Forest",
            **{key: rf_results[key] for key in comparison_metrics},
        },
        {
            "Model": "Logistic Regression",
            **{key: lr_results[key] for key in comparison_metrics},
        },
        {
            "Model": "Hybrid IDS",
            **{key: hybrid_results[key] for key in comparison_metrics},
        },
    ])

    comparison = comparison.rename(
        columns={
            "average_precision": "average_precision",
        }
    )

    comparison_path = REPORT_DIR / "model_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(f"Model comparison saved to: {comparison_path}")

    # ---------------------------------------------------------
    # 8. Error Analysis
    # ---------------------------------------------------------

    print("\n[8/8] Performing error analysis...")

    error_df = create_error_dataframe(
        X_test=X_test,
        y_test=y_test,
        rf_predictions=rf_predictions,
        lr_predictions=lr_predictions,
        hybrid_predictions=hybrid_predictions,
        rf_probabilities=rf_probabilities,
        lr_probabilities=lr_probabilities,
        hybrid_probabilities=hybrid_probabilities,
    )

    save_error_analysis(
        error_df,
        output_dir=REPORT_DIR,
    )

    # Hybrid error summary.

    hybrid_errors = error_df[
        error_df["Hybrid_Error"] == 1
    ]

    print("\nHybrid IDS Error Summary")
    print("-" * 45)

    print(f"Total test samples      : {len(error_df)}")
    print(f"Incorrect predictions   : {len(hybrid_errors)}")

    if "Hybrid_Error_Type" in hybrid_errors.columns:

        print("\nError types:")

        print(
            hybrid_errors["Hybrid_Error_Type"]
            .value_counts()
            .to_string()
        )
    # ---------------------------------------------------------
    # ATTACK-CATEGORY ANALYSIS
    # ---------------------------------------------------------

    print("\nPerforming attack-category analysis...")

    test_attack_categories = metadata["test_attack_categories"]

    predictions_by_model = {
        "Random Forest": rf_predictions,
        "Logistic Regression": lr_predictions,
        "Hybrid IDS": hybrid_predictions,
    }

    category_results, normal_results = analyze_attack_categories(
        y_true=y_test,
        attack_categories=test_attack_categories,
        predictions_by_model=predictions_by_model,
    )

    # ---------------------------------------------------------
    # KNOWN-VERSUS-UNSEEN ATTACK GENERALISATION ANALYSIS
    # ---------------------------------------------------------

    print("\\nPerforming known-versus-unseen attack generalisation analysis...")

    (
        known_unseen_results,
        individual_attack_results,
        category_generalization_results,
    ) = analyze_generalization(
        y_true=y_test,
        train_attack_labels=metadata["train_attack_labels"],
        test_attack_labels=metadata["test_attack_labels"],
        attack_categories=metadata["test_attack_categories"],
        predictions_by_model=predictions_by_model,
        output_dir=REPORT_DIR.parent,
    )

    print("Generalisation analysis completed.")
    # ---------------------------------------------------------
    # SHAP Explainability
    # ---------------------------------------------------------

    print("\nGenerating SHAP explanations...")

    try:

        explainer = SHAPExplainer(
            calibrated_rf,
            X_fit,
        )

        explainer.generate_all(
            X_test,
        )

        print("SHAP analysis completed.")

    except Exception as error:

        print(f"SHAP analysis could not be completed: {error}")

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED")
    print("=" * 65)

    print("\nFinal Hybrid IDS Performance")
    print("-" * 45)

    for metric, value in hybrid_results.items():

        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            print(f"{metric:25s}: {value}")

    print(f"\nReports and predictions saved in: {REPORT_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
