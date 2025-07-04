# Model Summary

## Model

- Name: XGBoost, version: 1.0.0
- Description: XGBoost model for Titanic survival prediction

### Parameters

- n_estimators: 100
- learning_rate: 0.01
- max_depth: 4
- use_label_encoder: False
- eval_metric: logloss
- random_state: 42

## Features

- Categorical: ['Pclass', 'Sex', 'Embarked']
- Numerical: ['Age', 'Fare', 'FamilySize']

## Preprocessing

- Fillna: {'Age': 'median', 'Fare': 'median', 'Embarked': 'mode'}
- Created Features: ['FamilySize']
- Dropped Features: ['SibSp', 'Parch', 'Name', 'Ticket', 'Cabin']

## Feature Engineering

- OneHot Features: ['Pclass', 'Sex', 'Embarked']
- OneHot Params: {'drop': 'first', 'sparse_output': False, 'handle_unknown': 'ignore'}
- Scaling: {'features': [], 'method': None}

## Training

- Train Size: 0.8
- Test Size: 0.2
- Metrics: ['accuracy']

## Output

- Model Path: `models/v1/model_xgb.pkl`
- Submission File: `data/submission/submission.csv`