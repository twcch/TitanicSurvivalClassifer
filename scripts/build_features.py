import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from core.data import load_train_processed_data, save_train_features_data
from core.pipeline.feature_engineering import build_features


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_processed_data = load_train_processed_data()

    train_features_data = build_features(train_processed_data, config)

    save_train_features_data(train_features_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    args = parser.parse_args()
    main(args.config)
