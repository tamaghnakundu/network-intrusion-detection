"""
Dataset Loader for Hybrid Intrusion Detection System
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    TRAIN_DATA,
    TEST_DATA,
    FEATURE_NAMES,
)


def _check_dataset_paths():
    """
    Verify that all required dataset files exist.
    """

    required_files = [
        TRAIN_DATA,
        TEST_DATA,
        FEATURE_NAMES,
    ]

    for file in required_files:
        if not Path(file).exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file}"
            )


def _load_feature_names():
    """
    Load feature names from feature_names.txt.
    """

    feature_names = (
        FEATURE_NAMES
        .read_text()
        .strip()
        .splitlines()
    )

    return feature_names


def _encode_categorical_columns(train_df, test_df):
    """
    Encode categorical columns using LabelEncoder.
    """

    categorical_columns = [
        "protocol_type",
        "service",
        "flag",
    ]

    encoders = {}

    for column in categorical_columns:

        encoder = LabelEncoder()

        train_df[column] = encoder.fit_transform(
            train_df[column]
        )

        test_df[column] = encoder.transform(
            test_df[column]
        )

        encoders[column] = encoder

    return train_df, test_df, encoders


def load_data():
    """
    Load and preprocess the NSL-KDD dataset.

    Returns
    -------
    train_df : pandas.DataFrame

    test_df : pandas.DataFrame

    feature_names : list
    """

    _check_dataset_paths()

    feature_names = _load_feature_names()

    train_df = pd.read_csv(
        TRAIN_DATA,
        names=feature_names,
    )

    test_df = pd.read_csv(
        TEST_DATA,
        names=feature_names,
    )

    # Convert labels to binary
    train_df["label"] = (
        train_df["label"] != "normal"
    ).astype(int)

    test_df["label"] = (
        test_df["label"] != "normal"
    ).astype(int)

    train_df, test_df, _ = _encode_categorical_columns(
        train_df,
        test_df,
    )

    return (
        train_df,
        test_df,
        feature_names,
    )