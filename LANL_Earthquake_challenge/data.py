from typing import Tuple
import pandas as pd
import numpy as np


class FeatureExtraction:
    TEST_SEGMENT_SIZE = 150_000

    def ___init__(self, ts_size: int):
        # Number of entries which make up one time stamp. note that features are learnt from this many datapoints
        self._ts_size = ts_size
        assert FeatureExtraction.TEST_SEGMENT_SIZE % self._ts_size == 0

    def get_y(self, df: pd.DataFrame) -> pd.Series:
        df = df[['time_to_failure']].copy()
        ts_count = df.shape[0] // self._ts_size
        df['ts'] = np.repeat(list(range(ts_count)), self._ts_size)
        output_df = df.groupby('ts').last()
        output_df.index.name = 'ts'
        return output_df['time_to_failure']

    def get_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Index is #example_id,#timestamp_id. Columns are features.
        """
        # TODO: this is where more features will come in.
        df = df[['acoustic_data']].copy()
        ts_count = df.shape[0] // self._ts_size
        df['ts'] = np.repeat(list(range(ts_count)), self._ts_size)
        output_df = df.groupby('ts').describe()['acoustic_data']
        output_df.columns.name = 'features'
        return output_df

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = self.get_X(df)
        y_df = self.get_y(df)
        return (X_df, y_df)

    def get_X_y_generator(self, fname: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
        chunk_size = FeatureExtraction.TEST_SEGMENT_SIZE
        for start_index in range(0, df.shape, chunk_size):
            X_df, y_df = self.get_X_y(df.iloc[start_index:(start_index + chunk_size)])
            yield (X_df, y_df)


class Data:
    def __init__(self, ts_window: int, ts_size: int):
        self._ts_window = ts_window
        self._ts_size = ts_size
        self._feature_extractor = FeatureExtraction(self._ts_size)

    def get_window_X(self, X_df: pd.DataFrame) -> np.array:
        row_count = X_df.shape[0] - self._ts_window + 1

        X = np.zeros(row_count, self._ts_window, X_df.shape[1])
        for i in range(self._ts_window, X_df.shape):
            X[i - self._ts_window] = X_df.values[i - self._ts_window:i, :]
        return X

    def get_window_y(self, y_df: pd.Series) -> np.array:
        return y_df.values[self._ts_window - 1:]

    def get_window_X_y(self, X_df, y_df) -> Tuple[np.array, np.array]:
        X = self.get_window_X(X_df)
        y = self.get_window_y(y_df)
        return (X, y)

    def get_X_y_generator(self, fname: str) -> Tuple[np.array, np.array]:
        gen = self._feature_extractor.get_X_y_generator(fname)
        for X_df, y_df in gen:
            X, y = self.get_window_X_y(X_df, y_df)
            yield (X, y)
