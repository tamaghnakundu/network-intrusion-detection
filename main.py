
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

from src.evaluation.attack_analysis import analyze_attack_categories
from src.evaluation.generalization_analysis import analyze_generalization
from src.explainability.shap_analysis import SHAPExplainer


def main():

    print("=" * 70)
    print("NETWORK INTRUSION DETECTION SYSTEM")
    print("=" * 70)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATASET
    # ---------------------------------------------------------

    print("\n[1/9] Loading NSL-KDD dataset...")

    X_train, y_train, X_test, y_test, metadata = (
        load_nsl_kdd(return_metadata=True)
    )

    print(f"Training samples  : {len(X_train)}")
    print(f"Testing samples   : {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    # ---------------------------------------------------------
    # 2. ATTACK DISTRIBUTION ANALYSIS
    # ---------------------------------------------------------

    print("\n[2/9] Analysing attack distribution...")

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

    print("\nTraining Category Distribution:")
    print(train_categories.value_counts())

    print("\nTesting Category Distribution:")
    print(test_categories.value_counts())

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

    attack_distribution.to_csv(
        REPORT_DIR / "attack_distribution.csv"
    )

    unseen_attacks = attack_distribution[
        (attack_distribution["Test_Samples"] > 0)
        & (attack_distribution["Train_Samples"] == 0)
    ]

    print("\nUnseen Attack Types:")
    print(unseen_attacks.to_string())

    print(
        f"\nNumber of unseen attack types: "
        f"{len(unseen_attacks)}"
    )

    # ---------------------------------------------------------
    # 3. SPLIT TRAINING DATA
    # ---------------------------------------------------------

    print("\n[3/9] Creating training and validation partitions...")

    X_fit, X_val, y_fit, y_val = split_training_data(
        X_train,
        y_train,
    )

    print(f"Model training samples: {len(X_fit)}")
    print(f"Validation samples    : {len(X_val)}")

    # ---------------------------------------------------------
    # 4. TRAIN MODELS
    # ---------------------------------------------------------

    print("\n[4/9] Training calibrated models...")

    calibrated_rf, calibrated_lr, calibrated_xgb = train_models(
        X_fit,
        y_fit,
    )

    models = {
        "rf": calibrated_rf,
        "lr": calibrated_lr,
        "xgb": calibrated_xgb,
    }

    print("Random Forest training completed.")
    print("Logistic Regression training completed.")
    print("XGBoost training completed.")

    # ---------------------------------------------------------
    # 5. OPTIMISE INDIVIDUAL MODEL THRESHOLDS
    # ---------------------------------------------------------

    print("\n[5/9] Optimising individual model thresholds...")

    individual_parameters = {}

    for model_name, model in models.items():

        probabilities = get_attack_probabilities(
            model,
            X_val,
        )

        parameters = optimize_threshold(
            y_val,
            probabilities,
        )

        individual_parameters[model_name] = parameters

        print(
            f"{model_name.upper()} threshold: "
            f"{parameters['threshold']:.3f}"
        )

    # ---------------------------------------------------------
    # 6. OPTIMISE ENSEMBLES
    # ---------------------------------------------------------

    print("\n[6/9] Optimising ensemble configurations...")

    ensemble_configurations = {
        "RF_LR": ["rf", "lr"],
        "RF_XGB": ["rf", "xgb"],
        "LR_XGB": ["lr", "xgb"],
        "RF_LR_XGB": ["rf", "lr", "xgb"],
    }

    ensemble_parameters = {}
    ensemble_validation_results = {}

    for ensemble_name, model_names in ensemble_configurations.items():

        print(f"\nOptimising {ensemble_name}...")

        selected_models = {
            name: models[name]
            for name in model_names
        }

        parameters = optimize_ensemble(
            models=selected_models,
            X_val=X_val,
            y_val=y_val,
        )

        ensemble_parameters[ensemble_name] = parameters

        ensemble_validation_results[ensemble_name] = {
            "validation_mcc": parameters["validation_mcc"],
            "validation_f1": parameters["validation_f1"],
            "validation_recall": parameters["validation_recall"],
        }

        print("Selected weights:", parameters["weights"])

        print(
            f"Selected threshold: "
            f"{parameters['threshold']:.3f}"
        )

        print(
            f"Validation MCC: "
            f"{parameters['validation_mcc']:.4f}"
        )

    # Select the best ensemble using validation results only.
    # KDDTest+ is not used for this selection.

    best_ensemble_name = max(
        ensemble_validation_results,
        key=lambda name: (
            ensemble_validation_results[name]["validation_mcc"],
            ensemble_validation_results[name]["validation_f1"],
            ensemble_validation_results[name]["validation_recall"],
        ),
    )

    print("\nBest ensemble selected using validation data:")
    print(best_ensemble_name)

    # ---------------------------------------------------------
    # SAVE OPTIMISATION PARAMETERS
    # ---------------------------------------------------------

    all_parameters = {
        "individual_models": individual_parameters,
        "ensembles": ensemble_parameters,
        "best_ensemble": best_ensemble_name,
        "validation_results": ensemble_validation_results,
    }

    parameters_path = REPORT_DIR / "optimisation_parameters.json"

    with open(parameters_path, "w", encoding="utf-8") as file:

        json.dump(
            all_parameters,
            file,
            indent=4,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else str(value)
            ),
        )

    print(f"\nParameters saved to: {parameters_path}")

    # ---------------------------------------------------------
    # 7. EVALUATE ON UNTOUCHED KDDTEST+
    # ---------------------------------------------------------

    print("\n[7/9] Evaluating models on KDDTest+...")

    predictions_by_model = {}
    probabilities_by_model = {}
    results_by_model = {}

    # Evaluate individual models.

    model_display_names = {
        "rf": "Random Forest",
        "lr": "Logistic Regression",
        "xgb": "XGBoost",
    }

    for model_name, model in models.items():

        probabilities = get_attack_probabilities(
            model,
            X_test,
        )

        threshold = individual_parameters[model_name]["threshold"]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        display_name = model_display_names[model_name]

        probabilities_by_model[display_name] = probabilities
        predictions_by_model[display_name] = predictions

        results_by_model[display_name] = evaluate_model(
            display_name,
            y_test,
            predictions,
            probabilities,
        )

        save_predictions(
            f"{model_name}_predictions.csv",
            y_test,
            predictions,
            probabilities,
        )

    # Evaluate all ensemble configurations.

    for ensemble_name, model_names in ensemble_configurations.items():

        selected_models = {
            name: models[name]
            for name in model_names
        }

        predictions, probabilities = predict_ensemble(
            models=selected_models,
            X=X_test,
            ensemble_parameters=ensemble_parameters[ensemble_name],
        )

        display_name = f"{ensemble_name} Ensemble"

        predictions_by_model[display_name] = predictions
        probabilities_by_model[display_name] = probabilities

        results_by_model[display_name] = evaluate_model(
            display_name,
            y_test,
            predictions,
            probabilities,
        )

        save_predictions(
            f"{ensemble_name.lower()}_predictions.csv",
            y_test,
            predictions,
            probabilities,
        )

    # ---------------------------------------------------------
    # 8. SAVE MODEL COMPARISON
    # ---------------------------------------------------------

    print("\n[8/9] Saving model comparison...")

    comparison_rows = []

    for model_name, results in results_by_model.items():

        row = {
            "Model": model_name,
            **results,
        }

        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nFinal Test Performance:")
    print(comparison.to_string(index=False))

    # ---------------------------------------------------------
    # ERROR ANALYSIS
    # ---------------------------------------------------------

    best_ensemble_display_name = (
        f"{best_ensemble_name} Ensemble"
    )

    error_df = create_error_dataframe(
        X_test=X_test,
        y_test=y_test,
        predictions_by_model=predictions_by_model,
        probabilities_by_model=probabilities_by_model,
    )

    save_error_analysis(
        error_df,
        output_dir=REPORT_DIR,
    )

    print("\nError analysis completed.")

    # ---------------------------------------------------------
    # ATTACK-CATEGORY ANALYSIS
    # ---------------------------------------------------------

    print("\nPerforming attack-category analysis...")

    analyze_attack_categories(
        y_true=y_test,
        attack_categories=test_categories,
        predictions_by_model=predictions_by_model,
    )

    print("Attack-category analysis completed.")

    # ---------------------------------------------------------
    # KNOWN-VERSUS-UNSEEN GENERALISATION ANALYSIS
    # ---------------------------------------------------------

    print(
        "\nPerforming known-versus-unseen "
        "generalisation analysis..."
    )

    analyze_generalization(
        y_true=y_test,
        train_attack_labels=metadata["train_attack_labels"],
        test_attack_labels=metadata["test_attack_labels"],
        attack_categories=test_categories,
        predictions_by_model=predictions_by_model,
        output_dir=REPORT_DIR.parent,
    )

    print("Generalisation analysis completed.")

    # ---------------------------------------------------------
    # SHAP EXPLAINABILITY
    # ---------------------------------------------------------

    print("\nGenerating SHAP explanations for Random Forest...")

    shap_completed = False

    try:

        explainer = SHAPExplainer(
            calibrated_rf,
            X_fit,
        )

        explainer.generate_all(X_test)

        shap_completed = True

        print("SHAP analysis completed.")

    except Exception as error:

        print(f"SHAP analysis failed: {error}")

        with open(
            REPORT_DIR / "shap_error.log",
            "w",
            encoding="utf-8",
        ) as file:

            file.write(repr(error))

    # ---------------------------------------------------------
    # 9. FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nBest ensemble selected using validation data:")
    print(best_ensemble_name)

    print("\nValidation metrics:")

    print(
        ensemble_validation_results[best_ensemble_name]
    )

    print("\nTest metrics for selected ensemble:")

    print(
        results_by_model[best_ensemble_display_name]
    )

    print(
        "\nSHAP analysis status:",
        "Completed" if shap_completed else "Failed",
    )

    print(f"\nReports saved in: {REPORT_DIR}")

    print("\nDone!")


if __name__ == "__main__":
    main()
