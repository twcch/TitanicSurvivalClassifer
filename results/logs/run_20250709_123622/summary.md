# Model Summary 

## Model

-    Name: xgboost, version: v1_0_0
-    Description: XGBoost model for Titanic survival prediction

### Parameters

-    n_estimators: 200
-    learning_rate: 0.01
-    max_depth: 10
-    use_label_encoder: False
-    eval_metric: logloss
-    random_state: 42

## Features

-    Categorical: None
-    Numerical: None

## Preprocessing

-    Fillna: {'Age': 'median', 'Fare': 'median', 'Embarked': 'mode'}
-    Created Features: []
-    Dropped Features: None

## Feature Engineering

-    OneHot Features: None
-    OneHot Params: None
-    Scaling: None

## Training

-    Train Size: None
-    Test Size: None
-    Metrics: None

## Output

-    Model Path: `results/v1_0_0/models/model.pkl`
-    Submission File: `results/v1_0_0/submission/submission.csv`