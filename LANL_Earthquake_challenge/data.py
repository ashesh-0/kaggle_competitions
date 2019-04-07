from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm


class FeatureExtraction:
    """
    Class is responsible for creating features from the data.
    """

    def __init__(self, ts_size: int, validation_fraction: float, segment_size: int = 150_000):
        # Number of entries which make up one time stamp. note that features are learnt from this many datapoints
        self._ts_size = ts_size
        self._segment_size = segment_size
        self._validation_fraction = validation_fraction

        assert self._segment_size % self._ts_size == 0
        assert self._validation_fraction >= 0

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
        output_df = output_df.drop('count', axis=1)
        return output_df

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = self.get_X(df)
        y_df = self.get_y(df)
        return (X_df, y_df)

    def get_validation_X_y(self, fname):
        if self._validation_fraction == 0:
            return (pd.DataFrame(), pd.Series())

        df = pd.read_csv(fname, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
        # We ensure that last few segments are not used in training data. We use it for validation.
        num_segments = df.shape[0] // self._segment_size
        validation_segments = int(self._validation_fraction * num_segments)
        return self.get_X_y(df.iloc[-1 * validation_segments * self._segment_size:])

    def get_X_y_generator(self, fname: str, padding_row_count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert padding_row_count % self._ts_size == 0

        df = pd.read_csv(fname, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

        # We ensure that last few segments are not used in training data. We use it for validation.
        num_segments = df.shape[0] // self._segment_size
        validation_segments = int(self._validation_fraction * num_segments)
        train_segments = num_segments - validation_segments

        print('Validation Size', validation_segments * self._segment_size / self._ts_size)
        print('Train Size', train_segments * self._segment_size / self._ts_size)

        next_first_index = 0
        for start_index in range(0, train_segments * self._segment_size, self._segment_size):
            padded_start_index = max(0, start_index - padding_row_count)
            X_df, y_df = self.get_X_y(df.iloc[padded_start_index:(start_index + self._segment_size)])
            X_df.index += next_first_index
            y_df.index += next_first_index

            # if it padding is non zero, few entries from last segment will come in next segment
            padded_first_entry_index = (1 + padding_row_count // self._ts_size)
            # the if-else is a corner case. If padding is more than segment_size then this will happen.
            if padded_first_entry_index <= y_df.shape[0]:
                next_first_index = y_df.index[-1 * padded_first_entry_index] + 1
            else:
                next_first_index = 0

            yield (X_df, y_df)


class Data:
    """
    This class uses FeatureExtraction class and creates a time sequence data.
    """

    def __init__(
            self,
            ts_window: int,
            ts_size: int,
            validation_fraction: float = 0.8,
            segment_size: int = 150_000,
    ):
        self._ts_window = ts_window
        self._ts_size = ts_size
        self._validation_fraction = validation_fraction
        self._feature_extractor = FeatureExtraction(
            self._ts_size,
            self._validation_fraction,
            segment_size=segment_size,
        )

    def get_window_X(self, X_df: pd.DataFrame) -> np.array:
        row_count = X_df.shape[0] - self._ts_window + 1

        X = np.zeros((row_count, self._ts_window, X_df.shape[1]))
        for i in range(self._ts_window, X_df.shape[0] + 1):
            X[i - self._ts_window] = X_df.values[i - self._ts_window:i, :]
        return X

    def get_window_y(self, y_df: pd.Series) -> np.array:
        return y_df.values[self._ts_window - 1:]

    def get_window_X_y(self, X_df, y_df) -> Tuple[np.array, np.array]:
        X = self.get_window_X(X_df)
        y = self.get_window_y(y_df)
        return (X, y)

    def get_validation_X_y(self, fname: str) -> Tuple[np.array, np.array]:
        """
        Returns last few segments of training data for validation. Note that this is not used in
        training, ie, this is not returned from get_X_y_generator()
        """

        X_df, y_df = self._feature_extractor.get_validation_X_y(fname)
        X, y = self.get_window_X_y(X_df, y_df)
        return (X, y)

    def get_X_y_generator(self, fname: str) -> Tuple[np.array, np.array]:
        # we need self._ts_window -1 rows at beginning to cater to starting data points in a chunk.
        padding = self._ts_size * (self._ts_window - 1)
        gen = self._feature_extractor.get_X_y_generator(fname, padding)
        for X_df, y_df in tqdm(gen):
            X, y = self.get_window_X_y(X_df, y_df)
            yield (X, y)


if __name__ == '__main__':
    ts_window = 100
    ts_size = 1000
    d = Data(ts_window, ts_size)
    gen = d.get_X_y_generator('train.csv')
    for X, y in gen:
        print('Shape of X', X.shape)
        print('Shape of y', y.shape)
