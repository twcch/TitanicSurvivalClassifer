# TitanicSurvivalPrediction

本專案為 Kaggle 經典入門題目「[Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)」的模組化實作。透過完整的資料前處理、特徵工程、模型訓練與評估流程，預測乘客在鐵達尼號災難中是否生存。

## Project Structure

TitanicSurvivalPrediction/
├── data/
│   ├── raw/  # 存放原始 data
│   ├── processed/  # 存放預處理後的 data
│   ├── features/  # 存放預處理後的 data
│   └── predictions/  # 存放模型預測後的 data
├── src/
│   ├── models/  # 各種模型實作
│   │   ├── linear_regression.py      
│   │   └── logistic_regression.py
│   ├── data/  # 資料讀寫與管理
│   ├── features/  # 特徵工程模組
│   └── __init__.py
├── notebooks/ #  分析用 notebook
│   ├── eda_raw.py
│   ├── eda_cleaned.py
│   └── __init__.py
├── scripts/  # 執行預處理、訓練、推論等腳本
│   ├── train_model.py  # 存放模型訓練的程式碼
│   ├── preprocess_data.py  # 
│   ├── build_features.py  # 
│   ├── inference.py  # 
│   └── __init__.py
├── models/  # 儲存訓練好的模型檔案
│   ├── v1/
│   │   ├── model_xgb.pkl  # 
│   │   └── config.json  # 儲存模型超參數設定
│   └── v2/
├── configs/  # yaml/json 儲存超參與版本設定
├── outputs/
│   ├── results/
│   │   └──  summary.md  # 記錄模型版本與指標
│   └── plots/
├── results/
│   └── summary.md  # 記錄模型版本與指標
├── requirements.txt
├── README.md
└── .gitignore

## 技術重點

-   資料清洗與缺失值處理
-   類別編碼與特徵工程
-   模型訓練 (eXtreme Gradient Boosting, XGBoost)
-   模型效能評估與推論

## 使用技術

-   Python 3.11+
-   Pandas, Numpy
-   Scikit-learn
-   XGBoost
-   Matplotlib / Seaborn

