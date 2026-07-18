"""
SHAP Explainability Module
"""

import matplotlib.pyplot as plt
import shap

from src.config import (
    PLOT_DIR,
    SHAP_SAMPLE_SIZE,
)


class SHAPExplainer:
    """
    SHAP explainability for the Random Forest model.
    """

    def __init__(
        self,
        model,
        X_train,
    ):
        """
        Initialize SHAP TreeExplainer.

        Parameters
        ----------
        model : RandomForestClassifier

        X_train : pandas.DataFrame
        """

        self.model = model
        self.X_train = X_train

        self.explainer = shap.TreeExplainer(model)

    def compute_shap_values(
        self,
        X_test,
        sample_size=SHAP_SAMPLE_SIZE,
    ):
        """
        Compute SHAP values on a subset of the test set.
        """

        X_sample = X_test.iloc[:sample_size].copy()

        shap_values = self.explainer(X_sample)

        return X_sample, shap_values

    def summary_bar(
        self,
        X_test,
        save=True,
    ):
        """
        SHAP Summary Bar Plot.
        """

        X_sample, shap_values = self.compute_shap_values(
            X_test
        )

        plt.figure()

        shap.plots.bar(
            shap_values[:, :, 1],
            show=False,
        )

        if save:

            PLOT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                PLOT_DIR / "shap_bar.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def beeswarm(
        self,
        X_test,
        save=True,
    ):
        """
        SHAP Beeswarm Plot.
        """

        X_sample, shap_values = self.compute_shap_values(
            X_test
        )

        plt.figure()

        shap.plots.beeswarm(
            shap_values[:, :, 1],
            show=False,
        )

        if save:

            PLOT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                PLOT_DIR / "shap_beeswarm.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def waterfall(
        self,
        X_test,
        sample_index=0,
        save=True,
    ):
        """
        SHAP Waterfall Plot for one sample.
        """

        X_sample, shap_values = self.compute_shap_values(
            X_test
        )

        plt.figure()

        shap.plots.waterfall(
            shap_values[sample_index, :, 1],
            show=False,
        )

        if save:

            PLOT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                PLOT_DIR / "shap_waterfall.png",
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def generate_all(
        self,
        X_test,
    ):
        """
        Generate all SHAP plots.
        """

        self.summary_bar(X_test)

        self.beeswarm(X_test)

        self.waterfall(X_test)