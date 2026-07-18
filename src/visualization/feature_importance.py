"""
Feature Importance Visualization
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    PLOT_DIR,
    TOP_FEATURES,
)


def plot_feature_importance(
    model_rf,
    feature_names,
    top_n=TOP_FEATURES,
    save=True,
):
    """
    Plot the top N most important features from the Random Forest model.

    Parameters
    ----------
    model_rf : RandomForestClassifier
        Trained Random Forest model.

    feature_names : list
        List of feature names.

    top_n : int
        Number of top features to display.

    save : bool
        Whether to save the figure.
    """

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model_rf.feature_importances_,
    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Top {top_n} Important Features"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    if save:

        PLOT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            PLOT_DIR / "feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def get_feature_importance(
    model_rf,
    feature_names,
):
    """
    Return feature importance as a sorted DataFrame.
    """

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model_rf.feature_importances_,
    })

    return (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def save_feature_importance(
    model_rf,
    feature_names,
):
    """
    Save feature importance values to CSV.
    """

    importance_df = get_feature_importance(
        model_rf,
        feature_names,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        PLOT_DIR / "feature_importance.csv",
        index=False,
    )