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

from pathlib import Path

import joblib

from xgboost import XGBClassifier


class XGBoostModel:
    def __init__(self, **kwargs):
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.01,
            "max_depth": 4,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.model = XGBClassifier(**default_params)
        self._artifacts = {}  # 用來儲存 encoder、scaler 等附加元件

    def fit(self, X, y):
        self.model.fit(X, y)

    def score(self, X, y):
        return self.model.score(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def set_artifact(self, name: str, obj) -> None:
        self._artifacts[name] = obj

    def get_artifact(self, name: str) -> object:
        return self._artifacts.get(name, None)

    def list_artifacts(self):
        return list(self._artifacts.keys())

    def load(self, path="models/v1/model_xgb.pkl"):
        path = Path(path)

        obj = joblib.load(path)
        self.model = obj["model"]
        self._artifacts = obj.get("artifacts", {})  # 如果有附加元件則載入

    def save(self, path="models/v1/model_xgb.pkl") -> None:
        save_path = Path(path)

        joblib.dump(
            {
                "model": self.model,
                "artifacts": self._artifacts
            },
            save_path,
        )
