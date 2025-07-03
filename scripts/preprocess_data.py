# -------------------------------------------------
# Copyright © 2025 Chih-Chien Hsieh 謝志謙
# All rights reserved.
#
# Github: https://github.com/twcch
# Website: https://twcch.io/
#
# This work is proprietary and confidential.
# No part of this codebase may be copied, modified, distributed, or used in any form without the prior written permission of the author.
# Unauthorized use is strictly prohibited and may result in legal consequences.
# -------------------------------------------------

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import pandas as pd
from core.data import load_raw_data, save_train_processed_data, save_test_processed_data

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"


def cast_data_types(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    for col in categorical_columns:
        if col in df.columns:
            # 將指定的類別欄位轉換為 category 類型
            df[col] = df[col].astype("category")
        else:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping type conversion.")

    return df


def fill_missing_values(df: pd.DataFrame, fillna_config: dict) -> pd.DataFrame:
    for col, method in fillna_config.items():
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            raise ValueError(f"Unsupported fillna method: {method} for column: {col}")

    return df


def create_features(df: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    for feature_name, detail in features_config.items():
        if detail["expression"] == "SibSp + Parch":
            df[feature_name] = df["SibSp"] + df["Parch"]
        else:
            raise ValueError(f"Unsupported feature expression: {detail['expression']} for feature: {feature_name}")

    return df


def drop_features(df: pd.DataFrame, drop_list: list) -> pd.DataFrame:
    df = df.drop(columns=drop_list)

    return df


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    df = cast_data_types(df, config["features"]["categorical"])
    df = fill_missing_values(df, config["preprocessing"]["fillna"])
    df = create_features(df, config["preprocessing"]["create_features"])
    df = drop_features(df, config["preprocessing"]["drop_features"])

    return df


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    train_raw, test_raw = load_raw_data()

    train_processed = preprocess(train_raw, config)
    test_processed = preprocess(test_raw, config)

    save_train_processed_data(train_processed)
    save_test_processed_data(test_processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1/config.json",
        help="Path to the configuration file",
    )

    args = parser.parse_args()
    main(args.config)
