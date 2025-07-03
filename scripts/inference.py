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
from core.data import load_test_features_data, save_submission
from core.models.xgboost import XGBoostModel

MODEL_PATH = "models/v1/model_xgb.pkl"
TEST_FEATURES_PATH = "data/features/test.csv"


def preprocess_test(test_data, encoder, config):
    categorical = config["features"]["categorical"]
    numerical = config["features"]["numerical"]

    if encoder is None:
        raise ValueError("Encoder not found in model artifacts.")

    X_cat = encoder.transform(test_data[categorical])
    X_all = np.hstack([X_cat, test_data[numerical].values])

    return X_all


def predict_and_save(model, X, passenger_ids, submission_path):
    predictions = model.predict(X)
    predictions = predictions.astype(int)

    df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})

    save_submission(df, submission_path)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    model = XGBoostModel()
    model.load(config["output"]["model_path"])
    encoder = model.get_artifact("encoder")
    test_data = load_test_features_data()
    X = preprocess_test(test_data, encoder, config)
    passenger_ids = test_data["PassengerId"]

    submission_path = config["output"]["submission_file"]
    predict_and_save(model, X, passenger_ids, submission_path)


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
