from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import logging


def csv_row_count(fname):
    line_count = 0
    with open(fname) as f:
        for line in f:
            line_count += 1

    gc.collect()
    # skip column names row
    return line_count - 1


class FeatureExtraction:
    """
    Class is responsible for creating features from the data.
    """

    def __init__(self, ts_size: int, validation_fraction: float, segment_size: int = 150_000, ls_batch_size=None):
        """
        Args:
            ts_size: how many data points become one timestamp.
            validation_fraction: fraction of data segments used for validation
            segment_size: number of datapoints to be considered one segment. it is same as batch_size*ts_size.
        """
        # Number of entries which make up one time stamp. note that features are learnt from this many datapoints
        self._ts_size = ts_size
        self._segment_size = segment_size
        # just for test. In prod, 100*self._segment_size is used.
        self._learn_scale_batch_size = ls_batch_size if ls_batch_size else 100 * self._segment_size

        self._validation_fraction = validation_fraction

        # to be set while fetching validation data/training data.
        # number of training examples to be given to model
        self.train_size = None
        # number of validation examples to be given to model
        self.validation_size = None
        # number of datapoints in training data file.
        self._raw_train_size = None

        # Used for normalization. Contains the maximum absolute value for a feature.
        self._scale_df = None

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

        if self._scale_df is not None:
            output_df = output_df / self._scale_df

        return output_df

    def learn_scale(self, fname):
        """
        Learns scale for normalization from training data. it does not touch validation data.
        """
        logging.info('Scale about to be learnt')

        self._scale_df = scale_df = None
        gen = self.get_X_y_generator(fname, 0, True, batch_size=self._learn_scale_batch_size)
        for X_df, _ in gen:
            max_df = X_df.abs().max()
            if scale_df is None:
                scale_df = max_df
            else:
                scale_df = pd.concat([scale_df, max_df], axis=1).max(axis=1)
            gc.collect()

        self._scale_df = scale_df

        logging.info('Scale learnt')

    def get_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X_df = self.get_X(df)
        y_df = self.get_y(df)
        return (X_df, y_df)

    def _set_train_validation_size(self):
        # We ensure that last few segments are not used in training data. We use it for validation.
        num_segments = self._raw_train_size // self._segment_size

        validation_segments = int(self._validation_fraction * num_segments)
        train_segments = num_segments - validation_segments

        self.train_size = int(train_segments * self._segment_size / self._ts_size)
        self.validation_size = int(validation_segments * self._segment_size / self._ts_size)
        print('Validation Size', self.validation_size)
        print('Train Size', self.train_size)

    def get_validation_X_y(self, fname):
        logging.info('Validation data requested.')
        if self._validation_fraction == 0:
            return (pd.DataFrame(), pd.Series())

        if self._raw_train_size is None:
            self._raw_train_size = csv_row_count(fname)
            self._set_train_validation_size()

        # read the column names
        dummy_df = pd.read_csv(
            fname,
            dtype={'acoustic_data': np.int16,
                   'time_to_failure': np.float64},
            nrows=3,
        )

        df = pd.read_csv(
            fname,
            dtype={'acoustic_data': np.int16,
                   'time_to_failure': np.float64},
            skiprows=self.train_size * self._ts_size,
        )

        df.columns = dummy_df.columns

        output_df = self.get_X_y(df)

        logging.info('Validation data returned.')
        return output_df

    def get_X_y_generator(self, fname: str, padding_row_count: int, test_mode: bool,
                          batch_size=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        test_mode: if True, then it returns entire training data in chunks only once. If False, it keeps
            returning chunks in cyclic order.
        """
        assert padding_row_count % self._ts_size == 0
        if batch_size is None:
            batch_size = self._segment_size

        if self._raw_train_size is None:
            self._raw_train_size = csv_row_count(fname)
            self._set_train_validation_size()

        while True:
            prev_df = None
            logging.info('Training X,y generator starting from beginning')
            # print('Training X,y generator starting from beginning')
            next_first_index = 0
            for df in pd.read_csv(
                    fname,
                    dtype={'acoustic_data': np.int16,
                           'time_to_failure': np.float64},
                    chunksize=batch_size,
            ):
                # We ensure that last few segments are not used in training data. We use it for validation.
                if next_first_index >= self.train_size:
                    break

                # assert next_first_index < self.train_size

                if prev_df is not None and padding_row_count > 0:
                    padded_df = pd.concat([prev_df.iloc[-1 * padding_row_count:], df])
                else:
                    padded_df = df

                X_df, y_df = self.get_X_y(padded_df)
                X_df.index += next_first_index
                y_df.index += next_first_index

                prev_df = df
                # if it padding is non zero, few entries from last segment will come in next segment
                padded_first_entry_index = (1 + padding_row_count // self._ts_size)
                # the if-else is a corner case. If padding is more than segment_size then this will happen.
                if padding_row_count == 0:
                    next_first_index = y_df.index[-1] + 1
                elif padded_first_entry_index <= y_df.shape[0]:
                    next_first_index = y_df.index[-1 * padded_first_entry_index + 1]
                else:
                    next_first_index = 0

                yield (X_df, y_df)
                gc.collect()

            if test_mode:
                break


class Data:
    """
    This class uses FeatureExtraction class and creates a time sequence data.
    """

    def __init__(
            self,
            ts_window: int,
            ts_size: int,
            train_fname: str,
            normalize: bool = True,
            validation_fraction: float = 0.2,
            segment_size: int = 150_000,
            ls_batch_size: int = None,
    ):
        self._ts_window = ts_window
        self._ts_size = ts_size
        self._validation_fraction = validation_fraction
        self._segment_size = segment_size
        self._train_fname = train_fname
        self._feature_extractor = FeatureExtraction(
            self._ts_size,
            self._validation_fraction,
            segment_size=segment_size,
            ls_batch_size=ls_batch_size,
        )

        self._normalize = normalize

        if self._normalize:
            # Learn scale.
            self._feature_extractor.learn_scale(self._train_fname)
            print('[Data] Scale learnt')

    def get_window_X(self, X_df: pd.DataFrame) -> np.array:
        row_count = X_df.shape[0] - self._ts_window + 1

        X = np.zeros((row_count, self._ts_window, X_df.shape[1]))
        for i in range(self._ts_window, X_df.shape[0] + 1):
            X[i - self._ts_window] = X_df.values[i - self._ts_window:i, :]
        return X

    def training_size(self):
        """
        Returns number of examples to be used in training.
        """
        return self._feature_extractor.train_size

    def batch_size(self):
        # 150_000
        return self._segment_size // self._ts_size

    def get_window_y(self, y_df: pd.Series) -> np.array:
        return y_df.values[self._ts_window - 1:]

    def get_window_X_y(self, X_df, y_df) -> Tuple[np.array, np.array]:
        X = self.get_window_X(X_df)
        y = self.get_window_y(y_df)
        return (X, y)

    def get_validation_X_y(self) -> Tuple[np.array, np.array]:
        """
        Returns last few segments of training data for validation. Note that this is not used in
        training, ie, this is not returned from get_X_y_generator()
        """

        X_df, y_df = self._feature_extractor.get_validation_X_y(self._train_fname)
        X, y = self.get_window_X_y(X_df, y_df)
        return (X, y)

    def get_X_y_generator(self, test_mode: bool = False) -> Tuple[np.array, np.array]:
        # We need self._ts_window -1 rows at beginning to cater to starting data points in a chunk.
        padding = self._ts_size * (self._ts_window - 1)
        gen = self._feature_extractor.get_X_y_generator(self._train_fname, padding, test_mode=test_mode)
        for X_df, y_df in tqdm(gen):
            X, y = self.get_window_X_y(X_df, y_df)
            yield (X, y)


if __name__ == '__main__':
    ts_window = 100
    ts_size = 1000
    d = Data(ts_window, ts_size)
    gen = d.get_X_y_generator('train.csv', test_mode=True)
    for X, y in gen:
        print('Shape of X', X.shape)
        print('Shape of y', y.shape)
