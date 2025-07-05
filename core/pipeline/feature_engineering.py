import pandas as pd


def create_features(data: pd.DataFrame, features_config: dict) -> pd.DataFrame:
    for feature_name, detail in features_config.items():
        if detail['expression'] == 'SibSp+Parch':
            data[feature_name] = data['SibSp'] + data['Parch']
        else:
            raise ValueError(f"Unsupported feature expression: {detail['expression']} for feature: {feature_name}")
    return data


def drop_features(data: pd.DataFrame, drop_list: list) -> pd.DataFrame:
    return data.drop(columns=[col for col in drop_list if col in data.columns])


def build_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = data.copy()
    data = create_features(data, config['feature_engineering']['create_features'])
    data = drop_features(data, config['feature_engineering']['drop_features'])
    return data
