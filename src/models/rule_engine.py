"""
Rule-Based Detection Module for Hybrid Intrusion Detection System
"""

import numpy as np

from src.config import RULE_PERCENTILE, RULE_MIN_SCORE


def compute_thresholds(X_train):
    """
    Compute feature thresholds using the specified percentile.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature set.

    Returns
    -------
    dict
        Dictionary containing thresholds for each rule.
    """

    thresholds = {
        "count": X_train["count"].quantile(RULE_PERCENTILE),
        "src_bytes": X_train["src_bytes"].quantile(RULE_PERCENTILE),
        "srv_count": X_train["srv_count"].quantile(RULE_PERCENTILE),
        "dst_host_srv_count":
            X_train["dst_host_srv_count"].quantile(RULE_PERCENTILE),
    }

    return thresholds


def rule_based_detector(sample, thresholds):
    """
    Apply rule-based detection to a single sample.

    Parameters
    ----------
    sample : pandas.Series

    thresholds : dict

    Returns
    -------
    int
        1 -> Attack
        0 -> Normal
    """

    score = 0

    if sample["count"] > thresholds["count"]:
        score += 1

    if sample["src_bytes"] > thresholds["src_bytes"]:
        score += 1

    if sample["srv_count"] > thresholds["srv_count"]:
        score += 1

    if sample["dst_host_srv_count"] > thresholds["dst_host_srv_count"]:
        score += 1

    return int(score >= RULE_MIN_SCORE)


def predict(X_test, thresholds):
    """
    Generate rule-based predictions.

    Parameters
    ----------
    X_test : pandas.DataFrame

    thresholds : dict

    Returns
    -------
    numpy.ndarray
    """

    predictions = np.array([
        rule_based_detector(row, thresholds)
        for _, row in X_test.iterrows()
    ])

    return predictions