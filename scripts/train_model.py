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
import numpy as np
import pandas as pd
import core.log_writer as lw
from sklearn.model_selection import train_test_split
from core.models.xgboost import XGBoostModel
from core.features.one_hot_feature_encoder import OneHotFeatureEncoder

FEATURES_PATH = "data/features/train.csv"
MODEL_PATH = "models/v1"


def load_data():
    df = pd.read_csv(FEATURES_PATH)

    return df


def preprocess_data(df: pd.DataFrame, config: dict):
    categorical = config["features"]["categorical"]
    numerical = config["features"]["numerical"]

    X = df[categorical + numerical]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # One-hot encoding for categorical features
    encoder_config = config["feature_engineering"]["encodings"]["onehot"]
    categorical = encoder_config["features"]
    encoder_params = encoder_config["params"]

    encoder = OneHotFeatureEncoder(
        categorical_features=categorical,
        **encoder_params
    )
    X_train_cat = encoder.fit_transform(X_train)
    X_test_cat = encoder.transform(X_test)

    X_train_all = np.hstack([X_train_cat, X_train[numerical].values])
    X_test_all = np.hstack([X_test_cat, X_test[numerical].values])

    return X_train_all, X_test_all, y_train, y_test, encoder


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    df = load_data()
    X_train_all, X_test_all, y_train, y_test, encoder = preprocess_data(df, config)

    model = XGBoostModel(**config["model"]["params"])
    model.fit(X_train_all, y_train)
    model.set_artifact("encoder", encoder)

    print("Training Accuracy:", model.score(X_train_all, y_train))
    print("Test Accuracy:", model.score(X_test_all, y_test))

    model.save(path=config["output"]["model_path"])

    lw.save_training_log(config, )


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
