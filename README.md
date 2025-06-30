# TitanicSurvivalPrediction

-   Kaggle url: https://www.kaggle.com/competitions/titanic

> 機器學習實戰專案｜模組化設計 × 設定檔驅動 × 結構清晰  
> 以 XGBoost 預測鐵達尼號乘客生存機率，展示資料工程與建模流程的完整掌握能力

## 專案亮點 Highlights

✅ 模組化架構: 依循業界慣例，分離 `data / features / models / utils` 等模組，利於擴充與維護。  
✅ 設定檔驅動: 使用 `config.json` 管理模型參數、特徵欄位與前處理規則，一鍵切換實驗設定。  
✅ 完整流程自動化: 從資料預處理、特徵建構、模型訓練到推論，全流程由 `main.py` 控制執行。  
✅ 可擴充日誌紀錄系統: 每次訓練自動產生 `logs/run_yyyymmdd_HHMMSS/`，儲存 config、metrics、summary。  
✅ 符合生產環境邏輯： 支援 `artifact` 儲存（如 encoder）、JSON 記錄模型設定與結果，便於部署與回溯。

---

## 🏗️ 專案結構 Project Structure

```bash
TitanicSurvivalPrediction/
├── configs/                  # 訓練與推論的 JSON 設定檔，集中管理模型參數、特徵欄位與前處理邏輯
│
├── data/
│   ├── raw/                 # 原始資料
│   ├── processed/           # 預處理後的資料
│   ├── features/            # 特徵工程後的資料
│   └── predictions/         # 模型預測結果
│
├── logs/
│   └── v1/                  # 每次訓練的執行記錄，自動產出含 config.json、summary.md 的日誌資料夾
│       └── run_YYYYMMDD_HHMMSS/
│           ├── config.json     # 當次訓練的設定檔快照
│           └── summary.md      # Markdown 格式總結
│
├── models/                  # 儲存訓練完成的模型物件與其對應設定
│   └── v1/
│       ├── model_xgb.pkl        # XGBoost 模型 (二進位格式，由 joblib 儲存)
│       └── config.json          # 與此模型對應的訓練參數與流程設定
│
├── notebooks/              # 用於探索性資料分析 (EDA) 的 Jupyter Notebook 檔案
│   ├── eda_raw.ipynb           # 原始資料的探索與視覺化
│   └── eda_cleaned.ipynb       # 清洗後資料的分析與確認
│
├── outputs/
│   └── plots/                  # 可視化圖片輸出，如特徵分布、模型重要性圖等
│
├── scripts/                 # 封裝好的功能性腳本，可單獨執行流程階段
│   ├── preprocess_data.py      # 資料預處理
│   ├── build_features.py       # 特徵工程
│   ├── train_model.py          # 模型訓練主程式，會輸出訓練模型與 log 記錄
│   └── inference.py            # 模型載入與測試集預測，輸出 submission 檔案
│
├── src/                    # 專案內部模組
│   ├── features/               # 特徵工程模組
│   │   └── one_hot_feature_encoder.py  # 包裝 sklearn 的 OneHotEncoder，含自定義 artifact 儲存邏輯
│   ├── models/                 # 模型封裝模組
│   │   └── xgboost.py          # 自訂 XGBoost 類別，支援 fit/predict/save/load/artifact 儲存
│   ├── data.py                # 資料載入與存檔的統一介面
│   ├── generate_summary.py    # 自動產出 Markdown 格式的模型摘要
│   └── log_writer.py          # 訓練 log 管理模組，將 config、metrics、summary 一起寫入 logs 資料夾
│
├── main.py                 # 主控腳本，串接整體流程
├── requirements.txt        # Python 環境依賴檔案，使用 pip install -r 安裝
├── README.md               # 專案說明文件
└── .gitignore              # Git 忽略追蹤清單
```

## 技術與套件 Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn
- xgboost
- joblib (模型儲存)
- pathlib, json (設定與日誌處理)

## 執行方式 How to Run

### 1️⃣ 安裝套件

```bash
pip install -r requirements.txt
```

### 2️⃣ 一鍵執行完整流程

```bash
python3 main.py
```

### 輸出結果

-   模型儲存於 `models/v1/model_xgb.pkl`
-   預測輸出於 `data/submission/submission.csv`
-   訓練紀錄自動寫入 `logs/run_YYYYMMDD_HHMMSS/`

## 訓練成果範例 (XGBoost)

| 指標             | 數值      |
|------------------|-----------|
| Training Accuracy | 0.844     |
| Test Accuracy     | 0.793     |

## 🧩 延伸功能建議 Extension Ideas

-   支援更多模型 (如 RandomForest、LogisticRegression)
-   加入交叉驗證、Grid Search、SHAP 模型解釋
-   加入標準化 (StandardScaler) 模組
-   將訓練與預測流程包成 CLI 工具或 API

## 🙋‍♂️ 關於作者

本專案由 [@twcch](https://github.com/twcch) 親自設計與實作，旨在展示資料分析師對機器學習、特徵工程與模型訓練流程的深刻理解與工程實踐能力。

如需進一步說明或合作洽詢，歡迎聯繫！

---