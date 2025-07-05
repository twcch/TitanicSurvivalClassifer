import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from core.data import load_train_raw_data, save_train_processed_data
from core.pipeline.preprocessing import preprocess_data


def main(config_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_raw_data = load_train_raw_data()

    train_processed_data = preprocess_data(train_raw_data, config)

    save_train_processed_data(train_processed_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    args = parser.parse_args()
    main(args.config)
