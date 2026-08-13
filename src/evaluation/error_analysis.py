import pandas as pd


def create_error_dataframe(
    X_test,
    y_test,
    rf_predictions,
    lr_predictions,
    rule_predictions,
    hybrid_predictions,
    confidence_scores,
):
    """
    Create a dataframe containing predictions from every model
    along with the ground-truth labels.
    """

    error_df = X_test.copy()

    error_df["Actual"] = y_test.values
    error_df["RandomForest"] = rf_predictions
    error_df["LogisticRegression"] = lr_predictions
    error_df["RuleEngine"] = rule_predictions
    error_df["Hybrid"] = hybrid_predictions

    error_df["Confidence"] = confidence_scores

    error_df["Correct"] = (
        error_df["Actual"] == error_df["Hybrid"]
    )

    return error_df

def get_misclassified_samples(error_df):
    """
    Return only the misclassified samples.
    """

    return error_df[~error_df["Correct"]].copy()

def split_errors(error_df):
    """
    Split errors into False Positives and False Negatives.
    """

    fp = error_df[
        (error_df["Actual"] == 0)
        & (error_df["Hybrid"] == 1)
    ].copy()

    fn = error_df[
        (error_df["Actual"] == 1)
        & (error_df["Hybrid"] == 0)
    ].copy()

    return fp, fn

def save_error_analysis(
    error_df,
    output_dir,
):
    """
    Save the complete prediction dataframe.
    """

    error_df.to_csv(
        output_dir / "error_analysis.csv",
        index=False,
    )