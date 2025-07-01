import json
from datetime import datetime
from pathlib import Path
from core.generate_summary import get_content

def create_log_dir(base_path="logs"):
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = Path(base_path) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def save_config(log_dir: Path, config: dict):
    path = log_dir / "config.json"
    path.write_text(json.dumps(config, indent=4), encoding="utf-8")


def save_metrics(log_dir: Path, metrics: dict):
    path = log_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")


def save_summary(log_dir: Path, summary_md: str):
    path = log_dir / "summary.md"
    path.write_text(summary_md, encoding="utf-8")


def save_training_log(config: dict, metrics=None, base_path="logs"):
    log_dir = create_log_dir(base_path)
    save_config(log_dir, config)
    #save_metrics(log_dir, metrics)
    save_summary(log_dir, get_content(config))
    print(f"\u2705 訓練記錄已儲存於：{log_dir}")
    return log_dir
