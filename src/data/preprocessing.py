"""
Data preprocessing for the Hybrid Intrusion Detection System.
"""

import pandas as pd


def prepare_data(train_df, test_df):
    """
    Prepare the training and test datasets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset.

    test_df : pandas.DataFrame
        Test dataset.

    Returns
    -------
    X_train : pandas.DataFrame
        Training features.

    X_test : pandas.DataFrame
        Test features.

    y_train : pandas.Series
        Training labels.

    y_test : pandas.Series
        Test labels.

    feature_names : list
        Names of the training features.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Remove columns not used for training
    X_train = train_df.drop(
        columns=["label", "difficulty"],
        errors="ignore",
    )

    X_test = test_df.drop(
        columns=["label", "difficulty"],
        errors="ignore",
    )

    y_train = train_df["label"]

    y_test = test_df["label"]

    # Fill missing values (if any)
    X_train = X_train.fillna(0)

    X_test = X_test.fillna(0)

    # Store the exact feature names used for training
    feature_names = X_train.columns.tolist()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )