import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def run(self):
        self._drop_duplicates()
        self._drop_id_column()
        self._one_hot_encode()
        self._replace_nulls()
        self._convert_to_float()
        self._scale_features()
        return self.df

    def _drop_duplicates(self):
        self.df = self.df.drop_duplicates()

    def _drop_id_column(self):
        if 'SK_ID_PREV' in self.df.columns:
            self.df = self.df.drop(columns=['SK_ID_PREV'])

    def _one_hot_encode(self):
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

    def _replace_nulls(self):
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(-999)

    def _convert_to_float(self):
        self.df = self.df.astype(float)

    def _scale_features(self):
        scaler = MinMaxScaler()
        if not np.all(np.isfinite(self.df.to_numpy())):
            raise ValueError("Non-finite values (NaN, inf) detected before scaling.")
        
        self.df[:] = scaler.fit_transform(self.df)

