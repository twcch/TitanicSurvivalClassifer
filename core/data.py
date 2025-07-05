from enum import Enum
from pathlib import Path
from typing import Tuple

import pandas as pd

RAW_DATA_PATH = Path('data/raw')
PROCESSED_DATA_PATH = Path('data/processed')
FEATURES_DATA_PATH = Path('data/features')


class FileName(Enum):
    TRAIN = 'train.csv'
    TEST = 'test.csv'


def load_train_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH / FileName.TRAIN.value)


def load_test_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH / FileName.TEST.value)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_raw_data(), load_test_raw_data()


def save_train_processed_data(data: pd.DataFrame) -> None:
    data.to_csv(PROCESSED_DATA_PATH / FileName.TRAIN.value, index=False)


def load_train_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH / FileName.TRAIN.value)


def save_test_processed_data(data: pd.DataFrame) -> None:
    data.to_csv(PROCESSED_DATA_PATH / FileName.TEST.value, index=False)


def load_test_processed_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH / FileName.TEST.value)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_processed_data(), load_test_processed_data()


def save_train_features_data(data: pd.DataFrame) -> None:
    data.to_csv(FEATURES_DATA_PATH / FileName.TRAIN.value, index=False)


def load_train_features_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DATA_PATH / FileName.TRAIN.value)


def save_test_features_data(data: pd.DataFrame) -> None:
    data.to_csv(FEATURES_DATA_PATH / FileName.TEST.value, index=False)


def load_test_features_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DATA_PATH / FileName.TEST.value)


def load_features_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_features_data(), load_test_features_data()


def save_submission_data(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False)
