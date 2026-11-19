
"""
SHAP Explainability Module for Network Intrusion Detection.

Explains the underlying Random Forest estimator inside
the calibrated Random Forest pipeline.

SHAP values are calculated on the transformed feature space
after applying the fitted preprocessing pipeline.

Important:
    This implementation explains the first fitted Random Forest
    estimator inside CalibratedClassifierCV. It does not directly
    explain the complete calibrated classifier or any weighted
    ensemble.
"""

import numpy as np
import matplotlib.pyplot as plt
import shap

from src.config import (
    PLOT_DIR,
    SHAP_SAMPLE_SIZE,
    RANDOM_STATE,
)


class SHAPExplainer:
    """
    SHAP explainability for a calibrated Random Forest model.

    The first fitted cross-validation member is selected from
    CalibratedClassifierCV.

    SHAP explanations are generated for its underlying Random
    Forest estimator, using data transformed by its fitted
    preprocessing pipeline.
    """

    def __init__(self, model, X_train=None):
        """
        Initialize the SHAP explainer.

        Parameters
        ----------
        model : CalibratedClassifierCV
            Trained calibrated Random Forest model.

        X_train : pandas.DataFrame, optional
            Training features. Retained for compatibility.
        """

        self.model = model
        self.X_train = X_train

        self._extract_fitted_components()

        self.explainer = shap.TreeExplainer(
            self.rf_estimator
        )

    def _extract_fitted_components(self):
        """
        Extract the fitted preprocessing pipeline and
        Random Forest estimator from CalibratedClassifierCV.

        The first fitted cross-validation member is selected.
        """

        if not hasattr(self.model, "calibrated_classifiers_"):
            raise ValueError(
                "The calibrated model must be fitted before "
                "initializing SHAPExplainer."
            )

        calibrated_members = self.model.calibrated_classifiers_

        if not calibrated_members:
            raise ValueError(
                "No fitted calibrated classifiers were found."
            )

        # Select the first fitted cross-validation member.
        calibrated_member = calibrated_members[0]

        # Support current and older scikit-learn attribute names.
        fitted_pipeline = getattr(
            calibrated_member,
            "estimator",
            None,
        )

        if fitted_pipeline is None:
            fitted_pipeline = getattr(
                calibrated_member,
                "base_estimator",
                None,
            )

        if fitted_pipeline is None:
            raise AttributeError(
                "Could not extract the fitted estimator from "
                "CalibratedClassifierCV."
            )

        if not hasattr(fitted_pipeline, "named_steps"):
            raise TypeError(
                "Expected the calibrated estimator to be "
                "a scikit-learn Pipeline."
            )

        required_steps = {
            "preprocessor",
            "classifier",
        }

        if not required_steps.issubset(
            fitted_pipeline.named_steps.keys()
        ):
            raise ValueError(
                "The fitted pipeline must contain "
                "'preprocessor' and 'classifier' steps."
            )

        self.pipeline = fitted_pipeline

        self.preprocessor = fitted_pipeline.named_steps[
            "preprocessor"
        ]

        self.rf_estimator = fitted_pipeline.named_steps[
            "classifier"
        ]

        self.classes_ = self.rf_estimator.classes_

        attack_indices = np.where(self.classes_ == 1)[0]

        if len(attack_indices) == 0:
            raise ValueError(
                "The Random Forest does not contain attack class 1."
            )

        self.attack_class_index = int(attack_indices[0])

    def _transform_data(self, X):
        """
        Apply the already-fitted preprocessing pipeline.

        Returns
        -------
        transformed : numpy.ndarray
            Dense transformed feature matrix.

        feature_names : list
            Names of the transformed features.
        """

        transformed = self.preprocessor.transform(X)

        # Convert sparse matrices to dense arrays.
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        try:
            feature_names = (
                self.preprocessor.get_feature_names_out()
            )

        except (AttributeError, ValueError):
            feature_names = [
                f"Feature_{i}"
                for i in range(transformed.shape[1])
            ]

        if len(feature_names) != transformed.shape[1]:
            raise ValueError(
                "Number of feature names does not match "
                "the transformed feature matrix."
            )

        return transformed, feature_names

    def compute_shap_values(
        self,
        X_test,
        sample_size=SHAP_SAMPLE_SIZE,
    ):
        """
        Compute SHAP values for a reproducible subset
        of the supplied test data.

        SHAP values are calculated for the attack class (1).

        Parameters
        ----------
        X_test : pandas.DataFrame
            Raw test features.

        sample_size : int
            Maximum number of samples to explain.

        Returns
        -------
        X_sample : pandas.DataFrame
            Selected raw samples.

        shap_explanation : shap.Explanation
            SHAP values for the attack class.
        """

        if sample_size <= 0:
            raise ValueError(
                "sample_size must be greater than zero."
            )

        if len(X_test) == 0:
            raise ValueError(
                "X_test cannot be empty."
            )

        n_samples = min(
            sample_size,
            len(X_test),
        )

        X_sample = X_test.sample(
            n=n_samples,
            random_state=RANDOM_STATE,
        )

        # Transform using the fitted preprocessing pipeline.
        X_transformed, feature_names = (
            self._transform_data(X_sample)
        )

        # Compute raw SHAP explanation.
        raw_explanation = self.explainer(
            X_transformed
        )

        shap_values = np.asarray(
            raw_explanation.values
        )

        base_values = np.asarray(
            raw_explanation.base_values
        )

        # Handle class-specific SHAP output.
        if shap_values.ndim == 3:

            shap_values = shap_values[
                :,
                :,
                self.attack_class_index,
            ]

        elif shap_values.ndim != 2:

            raise ValueError(
                "Unexpected SHAP output dimensions: "
                f"{shap_values.shape}"
            )

        # Select the corresponding attack-class base values.
        if base_values.ndim == 2:

            if base_values.shape[1] > self.attack_class_index:

                base_values = base_values[
                    :,
                    self.attack_class_index,
                ]

            elif base_values.shape[1] == 1:

                base_values = base_values[:, 0]

            else:

                raise ValueError(
                    "Unexpected SHAP base-value dimensions: "
                    f"{base_values.shape}"
                )

        if base_values.ndim not in (0, 1):

            raise ValueError(
                "Unexpected SHAP base-value dimensions: "
                f"{base_values.shape}"
            )

        # Verify dimensional consistency.
        if shap_values.shape[0] != X_transformed.shape[0]:

            raise ValueError(
                "SHAP sample count does not match input data."
            )

        if shap_values.shape[1] != X_transformed.shape[1]:

            raise ValueError(
                "SHAP feature count does not match "
                "transformed data."
            )

        shap_explanation = shap.Explanation(
            values=shap_values,
            base_values=base_values,
            data=X_transformed,
            feature_names=list(feature_names),
        )

        return X_sample, shap_explanation

    def _save_plot(self, filename, save=True):
        """
        Save and close the current SHAP plot.
        """

        if save:

            PLOT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                PLOT_DIR / filename,
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    def summary_bar(self, X_test, save=True):
        """
        Generate a global SHAP feature importance bar plot.

        Features are ranked by mean absolute SHAP value.
        """

        _, shap_values = self.compute_shap_values(
            X_test
        )

        shap.plots.bar(
            shap_values,
            max_display=20,
            show=False,
        )

        self._save_plot(
            "shap_bar.png",
            save=save,
        )

    def beeswarm(self, X_test, save=True):
        """
        Generate a SHAP beeswarm plot.

        Shows the distribution and direction of feature
        contributions across the selected samples.
        """

        _, shap_values = self.compute_shap_values(
            X_test
        )

        shap.plots.beeswarm(
            shap_values,
            max_display=20,
            show=False,
        )

        self._save_plot(
            "shap_beeswarm.png",
            save=save,
        )

    def waterfall(
        self,
        X_test,
        sample_index=0,
        save=True,
    ):
        """
        Generate a SHAP waterfall plot for one sample.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Raw test features.

        sample_index : int
            Position of the selected sample within
            the reproducibly sampled subset.
        """

        X_sample, shap_values = self.compute_shap_values(
            X_test
        )

        if not 0 <= sample_index < len(X_sample):

            raise IndexError(
                "sample_index is outside the selected SHAP sample."
            )

        shap.plots.waterfall(
            shap_values[sample_index],
            max_display=20,
            show=False,
        )

        self._save_plot(
            "shap_waterfall.png",
            save=save,
        )

    def generate_all(self, X_test):
        """
        Generate all SHAP visualizations.

        Outputs
        -------
        shap_bar.png
            Global feature importance.

        shap_beeswarm.png
            Feature attribution distribution.

        shap_waterfall.png
            Local explanation for one test sample.
        """

        print("\nGenerating SHAP explanations...")

        self.summary_bar(X_test)

        self.beeswarm(X_test)

        self.waterfall(X_test)

        print(
            f"SHAP plots saved to: {PLOT_DIR}"
        )
