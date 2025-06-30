# 🚢 TitanicSurvivalPrediction

> 機器學習實戰專案｜模組化設計 × 設定檔驅動 × 結構清晰  
> 以 XGBoost 預測鐵達尼號乘客生存機率，展示資料工程與建模流程的完整掌握能力。

---

## 🧠 專案亮點 Highlights

✅ **模組化架構**：依循業界慣例，分離 `data / features / models / utils` 等模組，利於擴充與維護。  
✅ **設定檔驅動**：使用 `config.json` 管理模型參數、特徵欄位與前處理規則，一鍵切換實驗設定。  
✅ **完整流程自動化**：從資料預處理、特徵建構、模型訓練到推論，全流程由 `main.py` 控制執行。  
✅ **可擴充日誌紀錄系統**：每次訓練自動產生 `logs/run_yyyymmdd_HHMMSS/`，儲存 config、metrics、summary。  
✅ **符合生產環境邏輯**：支援 `artifact` 儲存（如 encoder）、JSON 記錄模型設定與結果，便於部署與回溯。

---

## 🏗️ 專案結構 Project Structure

```bash
TitanicSurvivalPrediction/
├── configs/                    # 設定檔（模型、特徵、前處理等）
├── data/                       # 原始 / 處理後 / 特徵化 / 輸出資料
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── submission/
├── logs/                       # 訓練日誌（包含 config, metrics, summary.md）
├── models/                     # 儲存訓練好的模型（含 encoder）
├── scripts/                    # 各階段流程（preprocess, build_features, train, inference, summary）
├── src/                        # 專案核心模組（data, features, models, utils）
└── main.py                     # 主控腳本，串接整體流程
```

---

## ⚙️ 技術與套件 Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn
- xgboost
- joblib（模型儲存）
- pathlib, json（設定與日誌處理）

---

## 🚀 執行方式 How to Run

### 1️⃣ 安裝套件
```bash
pip install -r requirements.txt
```

### 2️⃣ 一鍵執行完整流程
```bash
python main.py
```

### 3️⃣ 輸出結果
- 模型儲存於 `models/v1/model_xgb.pkl`
- 預測輸出於 `data/submission/submission.csv`
- 訓練紀錄自動寫入 `logs/run_YYYYMMDD_HHMMSS/`

---

## 📊 訓練成果範例 (XGBoost)

| 指標             | 數值      |
|------------------|-----------|
| Training Accuracy | 0.844     |
| Test Accuracy     | 0.793     |

---

## 🧩 延伸功能建議 Extension Ideas

- 支援更多模型（如 RandomForest、LogisticRegression）
- 加入交叉驗證、Grid Search、SHAP 模型解釋
- 加入標準化（StandardScaler）模組
- 將訓練與預測流程包成 CLI 工具或 API

---

## 🙋‍♂️ 關於作者

本專案由 [@twcch](https://github.com/twcch) 親自設計與實作，旨在展示資料分析師對機器學習、特徵工程與模型訓練流程的深刻理解與工程實踐能力。

如需進一步說明或合作洽詢，歡迎聯繫！

---