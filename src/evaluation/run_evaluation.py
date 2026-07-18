from src.evaluation.metrics import evaluate_model


def evaluate_all_models(
    y_test,
    pred_rf,
    pred_lr,
):

    cm_rf = evaluate_model(
        "Random Forest",
        y_test,
        pred_rf,
    )

    cm_lr = evaluate_model(
        "Logistic Regression",
        y_test,
        pred_lr,
    )

    return (
        cm_rf,
        cm_lr,
    )