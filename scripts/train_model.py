import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import pandas as pd
from typing import Tuple
from core.data import load_train_features_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from core.pipeline.encoding import encode_and_merge_features
from core.models.xgboost_model import XGBoostModel
from core.log_writer import save_training_log


def resolve_output_paths(config: dict, version: str) -> None:
    output_config = config.get('output', {})
    for key, path in output_config.items():
        if isinstance(path, str):
            output_config[key] = path.replace("{version}", version)

    return output_config


def get_effective_features(data: pd.DataFrame, config: dict) -> list:
    features = config['variable']['features']
    drop_features = config.get('feature_engineering', {}).get('drop_features', [])

    # 先合併 config 中的欄位，再從實際存在欄位中過濾
    selected = features['categorical'] + features['numerical']
    selected = [col for col in selected if col not in drop_features and col in data.columns]

    return selected


def holdout_split(data: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    selected_features = get_effective_features(data, config)
    X = data[selected_features]
    y = data[config['variable']['target']]

    params = config['training']['holdout']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=params['train_size'],
                                                        test_size=params['test_size'],
                                                        random_state=params['random_state'])

    return X_train, X_test, y_train, y_test


def generate_kfold_splits(data: pd.DataFrame, config: dict):
    selected_features = get_effective_features(data, config)
    X = data[selected_features]
    y = data[config['variable']['target']]

    params = config['training']['k-fold']
    kf = KFold(
        n_splits=params['n_splits'],
        shuffle=params['shuffle'],
        random_state=params['random_state']
    )

    folds = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        folds.append((X_train, X_test, y_train, y_test))

    return folds


def get_model_from_config(config: dict):
    model_name = config['model']['name'].lower()
    model_params = config['model'].get('params', {})

    if model_name == 'xgboost':
        return XGBoostModel(**model_params)
    # elif model_name == "random_forest":
    #     return RandomForestModel(**model_params)
    # elif model_name == "logistic_regression":
    #     return LogisticRegressionModel(**model_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model(y_true, y_pred) -> dict:
    """計算準確率與報表，轉為 JSON 可序列化格式"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


def train_model(data: pd.DataFrame, config: dict):
    model = get_model_from_config(config)

    if config['training']['method'] == 'holdout':
        X_train, X_test, y_train, y_test = holdout_split(data, config)

        X_train_all, X_test_all, encoder = encode_and_merge_features(X_train, X_test, config)

        model.fit(X_train_all, y_train)
        # 儲存 encoder (僅在 holdout 有意義)
        model.set_artifact('encoder', encoder)
        # 預測與評估
        y_train_pred = model.predict(X_train_all)
        y_test_pred = model.predict(X_test_all)

        metrics = {
            "train": evaluate_model(y_train, y_train_pred),
            "test": evaluate_model(y_test, y_test_pred)
        }

    elif config['training']['method'] == 'k-fold':
        folds = generate_kfold_splits(data, config)
        fold_metrics = []

        for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
            print(f"[Fold {i + 1}] Training...")
            X_train_all, X_test_all, encoder = encode_and_merge_features(X_train, X_test, config)

            fold_model = get_model_from_config(config)  # 每折重建模型
            fold_model.fit(X_train_all, y_train)

            y_train_pred = fold_model.predict(X_train_all)
            y_test_pred = fold_model.predict(X_test_all)

            fold_metrics.append({
                "fold": i + 1,
                "train": evaluate_model(y_train, y_train_pred),
                "test": evaluate_model(y_test, y_test_pred)
            })

        metrics = fold_metrics  # 傳回整體每折評估結果
    else:
        raise ValueError("Unsupported training method")

    model.save(resolve_output_paths(config, config['model']['version'])['model_path'])

    return metrics


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_features_data = load_train_features_data()
    metrics = train_model(train_features_data, config)
    save_training_log(config, metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    args = parser.parse_args()
    main(args.config)
