"""
Hybrid IDS Module

Combines:
1. Random Forest
2. Logistic Regression
3. Rule-Based Detector
"""

import numpy as np

from src.config import (
    HYBRID_WEIGHTS,
    HYBRID_THRESHOLD,
)


class HybridIDS:
    """
    Weighted Hybrid Intrusion Detection System.
    """

    def __init__(self):

        self.rf_weight = HYBRID_WEIGHTS["rf"]
        self.lr_weight = HYBRID_WEIGHTS["lr"]
        self.rule_weight = HYBRID_WEIGHTS["rule"]

        self.threshold = HYBRID_THRESHOLD

    def predict_proba(
        self,
        model_rf,
        model_lr,
        X_test,
        X_test_scaled,
        rule_predictions,
    ):
        """
        Compute hybrid attack probability.
        """

        rf_prob = model_rf.predict_proba(X_test)[:, 1]

        lr_prob = model_lr.predict_proba(X_test_scaled)[:, 1]

        rule_prob = np.asarray(rule_predictions)

        hybrid_score = (

            self.rf_weight * rf_prob

            +

            self.lr_weight * lr_prob

            +

            self.rule_weight * rule_prob

        )

        return hybrid_score

    def predict(
        self,
        model_rf,
        model_lr,
        X_test,
        X_test_scaled,
        rule_predictions,
    ):
        """
        Generate hybrid predictions.
        """

        scores = self.predict_proba(
            model_rf,
            model_lr,
            X_test,
            X_test_scaled,
            rule_predictions,
        )

        predictions = (
            scores >= self.threshold
        ).astype(int)

        return predictions

    def confidence(
        self,
        hybrid_scores,
    ):
        """
        Compute confidence for every prediction.
        """

        confidence_scores = np.maximum(
            hybrid_scores,
            1 - hybrid_scores,
        )

        return confidence_scores

    def predict_with_confidence(
        self,
        model_rf,
        model_lr,
        X_test,
        X_test_scaled,
        rule_predictions,
    ):
        """
        Convenience function returning
        predictions, scores and confidence.
        """

        scores = self.predict_proba(
            model_rf,
            model_lr,
            X_test,
            X_test_scaled,
            rule_predictions,
        )

        predictions = (
            scores >= self.threshold
        ).astype(int)

        confidence = self.confidence(
            scores
        )

        return (
            predictions,
            scores,
            confidence,
        )