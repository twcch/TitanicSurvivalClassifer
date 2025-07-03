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

import subprocess


def run_step(description, command):
    print(f"Start: {description}")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"{description} 失敗")
        exit(1)

    print(f"End: {description}")


def main():
    config_path = f"configs/v1/config.json"

    steps = [
        ("scripts/preprocess_data.py", f"python scripts/preprocess_data.py --config {config_path}"),
        ("scripts/build_features.py", f"python scripts/build_features.py --config {config_path}"),
        ("scripts/train_model.py", f"python scripts/train_model.py --config {config_path}"),
        ("scripts/inference.py", f"python scripts/inference.py --config {config_path}")
    ]

    for desc, cmd in steps:
        run_step(desc, cmd)


if __name__ == "__main__":
    main()
