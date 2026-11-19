
"""
Data preprocessing for the Network Intrusion Detection System.

Uses ColumnTransformer to handle numerical and categorical
features separately.

Numerical features:
    - Median imputation
    - Standard scaling

Categorical features:
    - Most frequent imputation
    - One-hot encoding
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Categorical features in NSL-KDD
CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag",
]


def build_preprocessor(X):
    """
    Construct the preprocessing pipeline.

    Parameters
    ----------
    X : pandas.DataFrame
        Raw feature DataFrame.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Configured preprocessing pipeline.
    """

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X.columns
    ]

    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor
