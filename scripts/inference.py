import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import pandas as pd
import numpy as np
from core.data import load_test_raw_data, save_submission_data
from core.models.xgboost_model import XGBoostModel
from core.pipeline.preprocessing import preprocess_data
from core.pipeline.feature_engineering import build_features


# def resolve_output_paths(config: dict, version: str) -> dict:
#     output_config = config.get('output', {})
#     resolved = {}
#     for key, path in output_config.items():
#         if isinstance(path, str):
#             resolved[key] = path.replace('{version}', version)
#     return resolved

def resolve_output_paths(config: dict, version: str) -> None:
    output_config = config.get('output', {})
    for key, path in output_config.items():
        if isinstance(path, str):
            output_config[key] = path.replace("{version}", version)

    return output_config


def get_numerical_features(config: dict, available_columns: list) -> list:
    drop_list = config.get('feature_engineering', {}).get('drop_features', [])
    numerical_all = config['variable']['features']['numerical']
    return [col for col in numerical_all if col in available_columns and col not in drop_list]


def main(config_path: str):
    # 1. 讀取 Config 並解析輸出路徑
    with open(config_path, 'r') as f:
        config = json.load(f)

    version = config['model']['version']
    output_paths = resolve_output_paths(config, version)

    # 2. 載入與處理 Test Data
    test_raw = load_test_raw_data()
    test_processed = preprocess_data(test_raw, config)
    test_features = build_features(test_processed, config)

    # 3. 載入 Model 與 Encoder
    model = XGBoostModel.load(output_paths['model_path'])
    encoder = model.get_artifact('encoder')

    # 4. Feature Encoding
    categorical = config['training']['encoding']['onehot']['features']
    numerical_kept = get_numerical_features(config, test_features.columns)

    X_cat = encoder.transform(test_features)
    X_num = test_features[numerical_kept].values
    X_all = np.hstack([X_cat, X_num])

    # 5. 預測與提交
    y_pred = model.predict(X_all)
    submission = pd.DataFrame({
        "PassengerId": test_raw['PassengerId'],
        "Survived": y_pred.astype(int)
    })

    save_submission_data(submission, output_paths['submission_file'])
    print(f"✅ 預測完成，已儲存至: {output_paths['submission_file']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    main(args.config)
