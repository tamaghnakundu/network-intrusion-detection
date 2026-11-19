
"""
Dataset Loader for Network Intrusion Detection System.

Loads NSL-KDD and prepares binary classification labels
while preserving original attack labels and categories.

Binary classes:
    0 = Normal
    1 = Attack

Attack categories:
    DoS, Probe, R2L, U2R
"""

import pandas as pd

from src.config import TRAIN_DATA, TEST_DATA


# ---------------------------------------------------------
# Dataset Columns
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

LABEL_COLUMN = "label"
DIFFICULTY_COLUMN = "difficulty"

ALL_COLUMNS = (
    FEATURE_COLUMNS
    + [LABEL_COLUMN, DIFFICULTY_COLUMN]
)


# ---------------------------------------------------------
# NSL-KDD Attack Category Mapping
# ---------------------------------------------------------

ATTACK_CATEGORY_MAPPING = {

    # Denial of Service (DoS)
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "mailbomb": "DoS",
    "worm": "DoS",

    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",

    # Remote to Local (R2L)
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",

    # User to Root (U2R)
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}


# ---------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------

def _load_dataset(file_path):
    """
    Load an NSL-KDD dataset file.

    Parameters
    ----------
    file_path : Path
        Dataset file path.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        header=None,
        names=ALL_COLUMNS,
    )

    # Clean categorical and label columns.
    categorical_columns = [
        "protocol_type",
        "service",
        "flag",
        LABEL_COLUMN,
    ]

    for column in categorical_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


# ---------------------------------------------------------
# Binary Label Conversion
# ---------------------------------------------------------

def _convert_labels(labels):
    """
    Convert original NSL-KDD labels into binary classes.

    Normal = 0
    Attack = 1
    """

    labels = (
        labels.astype(str)
        .str.strip()
        .str.lower()
    )

    return (labels != "normal").astype(int)


# ---------------------------------------------------------
# Attack Category Conversion
# ---------------------------------------------------------

def _convert_attack_categories(labels):
    """
    Convert original attack labels into four attack categories.

    Normal traffic = Normal
    DoS attacks = DoS
    Probe attacks = Probe
    R2L attacks = R2L
    U2R attacks = U2R

    Unmapped attack names are labelled Unknown.
    """

    labels = (
        labels.astype(str)
        .str.strip()
        .str.lower()
    )

    categories = labels.map(ATTACK_CATEGORY_MAPPING)

    categories = categories.fillna("Unknown")

    categories = categories.where(
        labels != "normal",
        "Normal",
    )

    return categories


# ---------------------------------------------------------
# Main Dataset Loader
# ---------------------------------------------------------

def load_nsl_kdd(return_metadata=False):
    """
    Load and prepare NSL-KDD training and testing datasets.

    Parameters
    ----------
    return_metadata : bool, default=False
        If True, additionally return original attack labels,
        attack categories and difficulty levels.

    Returns
    -------
    If return_metadata=False:
        X_train, y_train, X_test, y_test

    If return_metadata=True:
        X_train, y_train, X_test, y_test, metadata
    """

    # Load datasets.
    train_df = _load_dataset(TRAIN_DATA)
    test_df = _load_dataset(TEST_DATA)

    # Separate original labels.
    train_attack_labels = train_df[LABEL_COLUMN].copy()
    test_attack_labels = test_df[LABEL_COLUMN].copy()

    # Preserve difficulty information.
    train_difficulty = train_df[DIFFICULTY_COLUMN].copy()
    test_difficulty = test_df[DIFFICULTY_COLUMN].copy()

    # Extract model features.
    X_train = train_df[FEATURE_COLUMNS].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()

    # Generate binary classification targets.
    y_train = _convert_labels(train_attack_labels)
    y_test = _convert_labels(test_attack_labels)

    # Preserve original four-value API by default.
    if not return_metadata:
        return (
            X_train,
            y_train,
            X_test,
            y_test,
        )

    # Generate attack categories.
    train_attack_categories = _convert_attack_categories(
        train_attack_labels
    )

    test_attack_categories = _convert_attack_categories(
        test_attack_labels
    )

    # Return additional metadata separately.
    metadata = {
        "train_attack_labels": train_attack_labels,
        "test_attack_labels": test_attack_labels,

        "train_attack_categories": train_attack_categories,
        "test_attack_categories": test_attack_categories,

        "train_difficulty": train_difficulty,
        "test_difficulty": test_difficulty,
    }

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata,
    )
