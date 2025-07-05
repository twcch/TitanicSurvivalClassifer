import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class OneHotFeatureEncoder:
    def __init__(self, categorical_features: list, **kwargs):
        self.categorical_features = categorical_features

        default_params = {
            'drop': 'first',
            'sparse_output': False,
            'handle_unknown': 'ignore'
        }

        default_params.update(kwargs)

        self.encoder = OneHotEncoder(**default_params)

        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        self.encoder.fit(data[self.categorical_features])
        self.is_fitted = True

        return self

    def transform(self, data: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError('Encoder must be fitted before transforming data.')

        X_cat = self.encoder.transform(data[self.categorical_features])

        if hasattr(X_cat, 'toarray'):  # for sparse matrix
            X_cat = X_cat.toarray()

        return X_cat

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def get_feature_names(self):
        if not self.is_fitted:
            raise RuntimeError('Encoder must be fitted before getting feature names.')

        return self.encoder.get_feature_names_out(self.categorical_features)
