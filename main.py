"""
Main entry point for the Hybrid Intrusion Detection System.
"""

from src.data.loader import load_data
from src.data.preprocessing import prepare_data

from src.models.random_forest import (
    train_random_forest,
    predict as rf_predict,
)

from src.models.logistic_regression import (
    train_logistic_regression,
    predict as lr_predict,
)

from src.models.rule_engine import (
    compute_thresholds,
    predict as rule_predict,
)

from src.models.hybrid import HybridIDS

from src.evaluation.metrics import (
    evaluate_model,
    save_predictions,
)

from src.visualization.confusion import (
    plot_all_confusion_matrices,
)

from src.visualization.feature_importance import (
    plot_feature_importance,
    save_feature_importance,
)

from src.explainability.shap_analysis import (
    SHAPExplainer,
)


def main():

    print("=" * 60)
    print("Hybrid Intrusion Detection System")
    print("=" * 60)

    ############################################################
    # Load Dataset
    ############################################################

    print("\nLoading dataset...")

    train_df, test_df, _ = load_data()

    ############################################################
    # Preprocess Data
    ############################################################

    print("Preprocessing dataset...")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = prepare_data(
        train_df,
        test_df,
    )   

    ############################################################
    # Random Forest
    ############################################################

    print("\nTraining Random Forest...")

    model_rf = train_random_forest(
        X_train,
        y_train,
    )

    rf_predictions = rf_predict(
        model_rf,
        X_test,
    )

    rf_results = evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
    )

    ############################################################
    # Logistic Regression
    ############################################################

    print("\nTraining Logistic Regression...")

    (
        model_lr,
        scaler,
        X_train_scaled,
        X_test_scaled,
    ) = train_logistic_regression(
        X_train,
        X_test,
        y_train,
    )

    lr_predictions = lr_predict(
        model_lr,
        X_test_scaled,
    )

    lr_results = evaluate_model(
        "Logistic Regression",
        y_test,
        lr_predictions,
    )

    ############################################################
    # Rule-Based Detector
    ############################################################

    print("\nRunning Rule-Based Detector...")

    thresholds = compute_thresholds(
        X_train,
    )

    rule_predictions = rule_predict(
        X_test,
        thresholds,
    )

    ############################################################
    # Hybrid IDS
    ############################################################

    print("\nRunning Hybrid IDS...")

    hybrid = HybridIDS()

    (
        hybrid_predictions,
        hybrid_scores,
        confidence_scores,
    ) = hybrid.predict_with_confidence(
        model_rf,
        model_lr,
        X_test,
        X_test_scaled,
        rule_predictions,
    )

    hybrid_results = evaluate_model(
        "Hybrid IDS",
        y_test,
        hybrid_predictions,
    )

    ############################################################
    # Save Predictions
    ############################################################

    save_predictions(
        "hybrid_predictions.csv",
        y_test,
        hybrid_predictions,
    )

    ############################################################
    # Feature Importance
    ############################################################

    print("\nGenerating Feature Importance...")

    plot_feature_importance(
        model_rf,
        feature_names,
    )

    save_feature_importance(
        model_rf,
        feature_names,
    )

    ############################################################
    # Confusion Matrices
    ############################################################

    print("\nGenerating Confusion Matrices...")

    plot_all_confusion_matrices({
        "Random Forest": rf_results,
        "Logistic Regression": lr_results,
        "Hybrid IDS": hybrid_results,
    })

    ############################################################
    # SHAP Explainability
    ############################################################

    print("\nGenerating SHAP Analysis...")

    explainer = SHAPExplainer(
        model_rf,
        X_train,
    )

    explainer.generate_all(
        X_test,
    )

    ############################################################
    # Confidence Statistics
    ############################################################

    print("\nPrediction Confidence")

    print("-" * 40)

    print(
        f"Average Confidence : {confidence_scores.mean():.4f}"
    )

    print(
        f"Minimum Confidence : {confidence_scores.min():.4f}"
    )

    print(
        f"Maximum Confidence : {confidence_scores.max():.4f}"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()