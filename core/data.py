# -------------------------------------------------
# Copyright © 2025 Chih-Chien Hsieh
# All rights reserved.
#
# Github: https://github.com/twcch
# Website: https://twcch.io/
#
# This work is proprietary and confidential.
# No part of this codebase may be copied, modified, distributed, or used in any form without the prior written permission of the author.
# Unauthorized use is strictly prohibited and may result in legal consequences.
# -------------------------------------------------

from enum import Enum
from pathlib import Path
from typing import Tuple

import pandas as pd

RAW_DATA_PATH = Path("data/raw/")
PROCESSED_DATA_PATH = Path("data/processed/")
FEATURES_DATA_PATH = Path("data/features/")


class FileName(Enum):
    TRAIN = "train.csv"
    TEST = "test.csv"
    SUBMISSION = "submission.csv"


def load_train_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH / FileName.TRAIN.value)


def load_test_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH / FileName.TEST.value)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = load_train_raw_data()
    test_data = load_test_raw_data()
    return train_data, test_data


def load_train_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH / FileName.TRAIN.value)


def save_train_processed_data(df: pd.DataFrame) -> None:
    file_path = PROCESSED_DATA_PATH / FileName.TRAIN.value

    df.to_csv(file_path, index=False)


def load_test_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH / FileName.TEST.value)


def save_test_processed_data(df: pd.DataFrame) -> None:
    file_path = PROCESSED_DATA_PATH / FileName.TEST.value

    df.to_csv(file_path, index=False)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = load_train_processed_data()
    test_data = load_test_processed_data()
    return train_data, test_data


def load_train_features_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DATA_PATH / FileName.TRAIN.value)


def save_train_features_data(df: pd.DataFrame) -> None:
    file_path = FEATURES_DATA_PATH / FileName.TRAIN.value

    df.to_csv(file_path, index=False)


def load_test_features_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DATA_PATH / FileName.TEST.value)


def save_test_features_data(df: pd.DataFrame) -> None:
    file_path = FEATURES_DATA_PATH / FileName.TEST.value

    df.to_csv(file_path, index=False)


def load_features_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = load_train_features_data()
    test_data = load_test_features_data()
    return train_data, test_data


def save_submission(df: pd.DataFrame, submission_path) -> None:
    df.to_csv(submission_path, index=False)
