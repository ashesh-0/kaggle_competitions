from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import logging

# logging.getLogger().setLevel(logging.DEBUG)


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

        self._train_X_dfs = []
        self._train_y_dfs = []

        assert self._segment_size % self._ts_size == 0
        assert self._validation_fraction >= 0

    def get_y(self, df: pd.DataFrame) -> pd.Series:
        df = df[['time_to_failure']].copy()
        ts_count = df.shape[0] // self._ts_size
        if df.shape[0] != ts_count * self._ts_size:
            logging.warn('For y, Trimming last {} entries'.format(df.shape[0] - ts_count * self._ts_size))
            df = df.iloc[:ts_count * self._ts_size]

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

        if df.shape[0] != ts_count * self._ts_size:
            logging.warn('For X, Trimming last {} entries'.format(df.shape[0] - ts_count * self._ts_size))
            df = df.iloc[:ts_count * self._ts_size]

        df['ts'] = np.repeat(list(range(ts_count)), self._ts_size)

        grp = df.groupby('ts')['acoustic_data']
        mean_df = grp.mean().to_frame('mean')
        std_df = grp.std().to_frame('std')
        quantile_df = grp.quantile([0.05, 0.5, 0.95]).unstack()
        quantile_df.columns = list(map(lambda x: 'Quantile-{}'.format(x), quantile_df.columns))
        max_df = grp.max().to_frame('max')
        min_df = grp.min().to_frame('min')
        output_df = pd.concat([mean_df, std_df, quantile_df, max_df, min_df], axis=1)
        output_df.columns.name = 'features'

        if self._scale_df is not None:
            output_df = output_df / self._scale_df

        gc.collect()
        return output_df

    def learn_scale_and_save_train_df(self, raw_data_df):
        """
        Learns scale for normalization from training data. it does not touch validation data.
        """
        logging.info('Scale about to be learnt')
        self._train_X_dfs = []
        self._train_y_dfs = []

        train_X_dfs = []
        train_y_dfs = []

        self._scale_df = scale_df = None
        gen = self.get_X_y_generator(raw_data_df, 0, batch_size=self._learn_scale_batch_size)

        for X_df, y_df in gen:
            gc.collect()
            train_X_dfs.append(X_df)
            train_y_dfs.append(y_df)

            max_df = X_df.abs().max()
            if scale_df is None:
                scale_df = max_df
            else:
                scale_df = pd.concat([scale_df, max_df], axis=1).max(axis=1)

        self._scale_df = scale_df

        logging.info('Scale learnt')
        self._train_X_dfs = [df / self._scale_df for df in train_X_dfs]
        self._train_y_dfs = train_y_dfs

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

    def get_validation_X_y(self, raw_data_df):
        logging.info('Validation data requested.')
        if self._validation_fraction == 0:
            return (pd.DataFrame(), pd.Series())

        if self._raw_train_size is None:
            self._raw_train_size = raw_data_df.shape[0]
            self._set_train_validation_size()

        val_Xs = []
        val_ys = []
        for start_index in range(self.train_size * self._ts_size, raw_data_df.shape[0], self._segment_size):
            df = raw_data_df.iloc[start_index:start_index + self._segment_size]
            val_X_df, val_y_df = self.get_X_y(df)
            val_Xs.append(val_X_df)
            val_ys.append(val_y_df)

        val_X_df = pd.concat(val_Xs)
        val_y_df = pd.concat(val_ys)

        logging.info('Validation data returned.')
        return (val_X_df, val_y_df)

    def get_X_y_generator_fast(self, padding_row_count, debug_mode=False):
        assert self._learn_scale_batch_size % self._segment_size == 0
        assert padding_row_count % self._ts_size == 0
        assert len(self._train_X_dfs) > 0 or self._raw_train_size < self._ts_size

        padding_row_count = padding_row_count // self._ts_size
        segment_size_row_count = self._segment_size // self._ts_size

        while True:
            prev_X_df = None
            prev_y_df = None
            for X_chunk_df, y_chunk_df in zip(self._train_X_dfs, self._train_y_dfs):
                for start_index in range(0, X_chunk_df.shape[0], segment_size_row_count):
                    padding_start_index = start_index - padding_row_count

                    if padding_start_index >= 0:
                        X_df = X_chunk_df.iloc[padding_start_index:start_index + segment_size_row_count]
                        y_df = y_chunk_df.iloc[padding_start_index:start_index + segment_size_row_count]
                        prev_X_df = X_df
                        prev_y_df = y_df
                        yield (X_df, y_df)
                    else:
                        X_df = X_chunk_df.iloc[:start_index + segment_size_row_count]
                        y_df = y_chunk_df.iloc[:start_index + segment_size_row_count]

                        if prev_X_df is not None:
                            X_df = pd.concat([prev_X_df.iloc[padding_start_index:], X_df])
                            y_df = pd.concat([prev_y_df.iloc[padding_start_index:], y_df])

                        prev_X_df = X_df
                        prev_y_df = y_df
                        yield (X_df, y_df)

            if debug_mode:
                break

    def get_X_y_generator(self, raw_data_df, padding_row_count: int,
                          batch_size=None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert padding_row_count % self._ts_size == 0
        if batch_size is None:
            batch_size = self._segment_size

        if self._raw_train_size is None:
            self._raw_train_size = raw_data_df.shape[0]
            self._set_train_validation_size()

        logging.info('Training X,y generator starting from beginning')
        next_first_index = 0
        for start_index in range(0, self.train_size * self._ts_size, batch_size):
            df = raw_data_df.iloc[max(0, start_index - padding_row_count):start_index + batch_size]

            X_df, y_df = self.get_X_y(df)
            X_df.index += next_first_index
            y_df.index += next_first_index
            gc.collect()

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
        self.raw_data_df = pd.read_csv(
            train_fname,
            dtype={'acoustic_data': np.int16,
                   'time_to_failure': np.float32},
        )

        self._feature_extractor = FeatureExtraction(
            self._ts_size,
            self._validation_fraction,
            segment_size=segment_size,
            ls_batch_size=ls_batch_size,
        )

        self._normalize = normalize

        if self._normalize:
            # Learn scale.
            self._feature_extractor.learn_scale_and_save_train_df(self.raw_data_df)
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

        X_df, y_df = self._feature_extractor.get_validation_X_y(self.raw_data_df)
        X, y = self.get_window_X_y(X_df, y_df)
        return (X, y)

    def get_test_X(self, df) -> np.array:
        """
        For Test data, it fetches data from file and returns the X with shape
             (#examples, self._ts_window, feature_count)
        """
        gc.collect()
        X_df = self._feature_extractor.get_X(df)
        return self.get_window_X(X_df)

    def get_X_y_generator(self, debug_mode: bool = False) -> Tuple[np.array, np.array]:
        # We need self._ts_window -1 rows at beginning to cater to starting data points in a chunk.
        padding = self._ts_size * (self._ts_window - 1)
        # gen = self._feature_extractor.get_X_y_generator(self.raw_data_df, padding, test_mode=test_mode)
        gen = self._feature_extractor.get_X_y_generator_fast(padding, debug_mode=debug_mode)

        for X_df, y_df in gen:
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
