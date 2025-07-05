from typing import Tuple

import numpy as np
import pandas as pd

from core.features.one_hot_feature_encoder import OneHotFeatureEncoder


def get_effective_numerical_columns(X: pd.DataFrame, config: dict) -> list:
    """依據 config 和 X 的實際欄位，取得有效的 numerical 欄位"""
    numerical_all = config['variable']['features']['numerical']
    drop_features = config.get('feature_engineering', {}).get('drop_features', [])

    numerical_kept = [col for col in numerical_all if col not in drop_features and col in X.columns]
    return numerical_kept


def encode_and_merge_features(X_train: pd.DataFrame, X_test: pd.DataFrame, config: dict) -> Tuple[
    np.ndarray, np.ndarray, OneHotFeatureEncoder]:
    """
    對 categorical features 進行 One-Hot Encoding，並與有效 numerical features 合併。
    回傳訓練與測試資料的特徵矩陣、以及 encoder 實體。
    """
    encoder_config = config['training']['encoding']['onehot']
    encoder_features = encoder_config['features']
    encoder_params = encoder_config['params']

    encoder = OneHotFeatureEncoder(
        categorical_features=encoder_features,
        **encoder_params
    )

    # One-hot encoding
    X_train_cat = encoder.fit_transform(X_train)
    X_test_cat = encoder.transform(X_test)

    # 處理 numerical 欄位（保留有效的欄位）
    numerical_kept = get_effective_numerical_columns(X_train, config)
    X_train_num = X_train[numerical_kept].values
    X_test_num = X_test[numerical_kept].values

    # 合併編碼後特徵
    X_train_all = np.hstack([X_train_cat, X_train_num])
    X_test_all = np.hstack([X_test_cat, X_test_num])

    return X_train_all, X_test_all, encoder
