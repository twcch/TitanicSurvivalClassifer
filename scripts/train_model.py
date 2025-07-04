import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import numpy as np
import pandas as pd
import core.log_writer as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from core.models.xgboost_model import XGBoostModel
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

    y_train_pred = model.predict(X_train_all)
    train_acc_score = accuracy_score(y_train, y_train_pred)
    train_conf_matrix = confusion_matrix(y_train, y_train_pred).tolist()  # 轉為 list 才能 JSON 化
    train_class_report = classification_report(y_train, y_train_pred, output_dict=True)  # 轉為 dict

    y_test_pred = model.predict(X_test_all)
    test_acc_score = accuracy_score(y_test, y_test_pred)
    test_conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()  # 轉為 list 才能 JSON 化
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)  # 轉為 dict

    metrics_summary = {
        "train": {
            "accuracy": train_acc_score,
            "confusion_matrix": train_conf_matrix,
            "classification_report": train_class_report
        },
        "test": {
            "accuracy": test_acc_score,
            "confusion_matrix": test_conf_matrix,
            "classification_report": test_class_report
        }
    }

    model.save(path=config["output"]["model_path"])

    lw.save_training_log(config, metrics_summary)


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
