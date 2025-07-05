import subprocess
import time
from datetime import datetime


def run_step(desc: str, cmd: str) -> None:
    print(f'🟢 [{datetime.now():%Y-%m-%d %H:%M:%S}] 開始執行：{desc}')
    print(f'↪️ 指令：{cmd}')

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    end_time = time.time()

    duration = end_time - start_time

    if result.returncode != 0:
        print(f'❌ [{desc}] 失敗 (耗時: {duration:.2f} 秒)\n')
        exit(1)

    print(f'✅ [{datetime.now():%Y-%m-%d %H:%M:%S}] 完成: {desc} (耗時: {duration:.2f} 秒)\n')


def main():
    version = 'v1_0_0'
    config_path = f'configs/{version}/config.json'

    steps = [
        # ('scripts/preprocess_data.py', f'python3 scripts/preprocess_data.py --config {config_path}'),
        # ('scripts/build_features.py', f'python3 scripts/build_features.py --config {config_path}'),
        # ('scripts/train_model.py', f'python3 scripts/train_model.py --config {config_path}'),
        ('scripts/inference.py', f'python3 scripts/inference.py --config {config_path}')
    ]

    for desc, cmd in steps:
        run_step(desc, cmd)


if __name__ == '__main__':
    main()
