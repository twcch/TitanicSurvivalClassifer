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

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
import pandas as pd
from core.data import load_processed_data, save_train_features_data, save_test_features_data

PROCESSED_PATH = "data/processed/"
FEATURES_PATH = "data/features/"


def build_features(df: pd.DataFrame, config: dict):
    df = df.copy()

    # 暫無 ...

    return df


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    train_processed, test_processed = load_processed_data()
    train_features = build_features(train_processed, config)
    test_features = build_features(test_processed, config)

    save_train_features_data(train_features)
    save_test_features_data(test_features)


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
