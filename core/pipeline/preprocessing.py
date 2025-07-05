import pandas as pd


def cast_category_type(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    for column in categorical_columns:
        if column in data.columns:
            data[column] = data[column].astype('category')
        else:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping type conversion.")
    return data


def fill_missing_values(data: pd.DataFrame, fillna_config: dict) -> pd.DataFrame:
    for column, method in fillna_config.items():
        if method == 'mean':
            data[column] = data[column].fillna(data[column].mean())
        elif method == 'median':
            data[column] = data[column].fillna(data[column].median())
        elif method == 'mode':
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            raise ValueError(f"Unsupported fillna method: {method} for column: {column}")
    return data


def preprocess_data(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    data = data.copy()
    data = cast_category_type(data, config['variable']['features']['categorical'])
    data = fill_missing_values(data, config['preprocessing']['fillna'])
    return data
