from unittest.mock import patch
import pandas as pd
from data import FeatureExtraction, Data
import numpy as np

test_df = pd.DataFrame(
    [
        [7, 2.5],
        [10, 1.5],
        [15, 5],
        [12, 7.5],
        [13, 12.5],
        [10, 1],
        [12, 5],
        [20, 7.5],
        [10, 2.5],
        [11, 5],
        [2, 7.5],
        [10, 2.5],
    ],
    columns=['acoustic_data', 'time_to_failure'],
)


def mock_read_csv(*args, **kwargs):
    return test_df


# def mock_csv_row_count(fname):
#     return test_df.shape[0]


class TestFeatureExtraction:
    def test_get_X_y(self):
        ts_size = 3
        validation_fraction = 0
        ex = FeatureExtraction(ts_size, validation_fraction)
        X_df, y_df = ex.get_X_y(test_df)

        assert X_df.shape[0] == y_df.shape[0]
        assert y_df.shape[0] == test_df.shape[0] // 3

        expected_y = test_df['time_to_failure'].shift(-1 * ts_size + 1)[::ts_size]
        expected_y.index = list(range(test_df.shape[0] // ts_size))
        assert y_df.equals(expected_y)

        assert X_df['mean_0->100'].iloc[0] == test_df['acoustic_data'].iloc[:ts_size].mean()
        assert X_df['mean_0->100'].iloc[1] == test_df['acoustic_data'].iloc[ts_size:2 * ts_size].mean()
        assert X_df['mean_0->100'].iloc[2] == test_df['acoustic_data'].iloc[2 * ts_size:3 * ts_size].mean()
        assert X_df['mean_0->100'].iloc[3] == test_df['acoustic_data'].iloc[3 * ts_size:4 * ts_size].mean()

    def test_get_X_should_compute_using_subset_data(self):
        """
        Within one timestamp chunk, this test ensures that features are computing using correct data subsets.
        """
        ts_size = 4
        validation_fraction = 0
        ex = FeatureExtraction(ts_size, validation_fraction)
        X_df = ex.get_X(test_df)
        for idx, start_index in enumerate(range(0, test_df.shape[0], ts_size)):
            df = test_df.iloc[start_index:start_index + ts_size]['acoustic_data']
            assert X_df['mean_0->100'].iloc[idx] == df.mean()
            assert X_df['mean_0->25'].iloc[idx] == df.iloc[:ts_size // 4].mean()
            assert X_df['mean_25->50'].iloc[idx] == df.iloc[ts_size // 4:2 * ts_size // 4].mean()
            assert X_df['mean_50->75'].iloc[idx] == df.iloc[2 * ts_size // 4:3 * ts_size // 4].mean()
            assert X_df['mean_75->100'].iloc[idx] == df.iloc[3 * ts_size // 4:4 * ts_size // 4].mean()

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_X_y_generator(self, mock_read):
        ts_size = 2
        segment_size = 4
        padding = 0
        validation_fraction = 0
        ex = FeatureExtraction(ts_size, validation_fraction, segment_size=segment_size)
        gen = ex.get_X_y_generator(test_df, padding)

        expected_y = test_df['time_to_failure'].shift(-1 * ts_size + 1)[::ts_size]
        expected_y.index = list(range(test_df.shape[0] // ts_size))
        num_ts_in_segment = segment_size // ts_size

        for i, data in enumerate(gen):
            X_df, y_df = data
            assert y_df.shape[0] == X_df.shape[0] == segment_size / ts_size
            assert expected_y[i * num_ts_in_segment:(i + 1) * num_ts_in_segment].equals(y_df)

        assert (i + 1) == test_df.shape[0] / segment_size

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_X_y_generator_with_padding(self, mock_read):
        ts_size = 2
        segment_size = 6
        padding = 1
        validation_fraction = 0
        ex = FeatureExtraction(
            ts_size,
            validation_fraction=validation_fraction,
            segment_size=segment_size,
        )
        gen = ex.get_X_y_generator(test_df, padding * ts_size)

        expected_y = test_df['time_to_failure'].shift(-1 * ts_size + 1)[::ts_size]
        expected_y.index = list(range(test_df.shape[0] // ts_size))
        num_ts_in_segment = segment_size // ts_size

        for i, data in enumerate(gen):
            X_df, y_df = data
            # In first segment, no way to have padding
            if i == 0:
                assert y_df.shape[0] == X_df.shape[0] == (segment_size / ts_size)
            else:
                assert y_df.shape[0] == X_df.shape[0]
                assert y_df.shape[0] == (segment_size / ts_size + padding)

            assert expected_y[max(0, i * num_ts_in_segment - padding):(i + 1) * num_ts_in_segment].equals(y_df)

        assert (i + 1) == test_df.shape[0] / segment_size


class TestData:
    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_window_X_y(self, mock_read):
        ts_size = 2
        segment_size = 6
        ts_window = 3
        validation_fraction = 0
        fe = FeatureExtraction(ts_size, validation_fraction, segment_size=segment_size)
        X_df, y_df = fe.get_X_y(test_df)
        dt = Data(ts_window, ts_size, '', validation_fraction=0, segment_size=segment_size, normalize=False)
        X, y = dt.get_window_X_y(X_df, y_df)
        assert X.shape == (X_df.shape[0] - ts_window + 1, ts_window, X_df.shape[1])
        assert y.shape == (X_df.shape[0] - ts_window + 1, )

        for i in range(X.shape[0]):
            assert y_df.values[i + ts_window - 1] == y[i]
            for j in range(ts_window):
                assert all(np.nan_to_num(X[i, j, :]) == np.nan_to_num(X_df.values[i + j, :]))

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_X_y_generator(self, mock_read):
        """
        Tests that whether one computes X,y in one go or computes through generator, resulting data is the same.
        """
        ts_size = 2
        segment_size = 6
        ts_window = 3
        validation_fraction = 0
        fe = FeatureExtraction(ts_size, validation_fraction, segment_size=segment_size)

        # unnormalized X_df
        X_df, y_df = fe.get_X_y(test_df)
        scale_df = X_df.abs().max()
        # normalized X_df. this should match with what comes out of generator.
        X_df = X_df / scale_df

        debug_mode = True
        dt = Data(
            ts_window,
            ts_size,
            '',
            validation_fraction=0,
            segment_size=segment_size,
            normalize=True,
            ls_batch_size=segment_size,
        )
        # get X and y from whole data.
        X_whole, y_whole = dt.get_window_X_y(X_df, y_df)

        # now X and y are fetched from generator and we ensure that it matches completely with above data.
        gen = dt.get_X_y_generator(debug_mode=debug_mode)
        last_index = 0
        for X, y in gen:
            assert (np.nan_to_num(X) == np.nan_to_num(X_whole[last_index:last_index + X.shape[0]])).all()
            assert (y == y_whole[last_index:last_index + y.shape[0]]).all()
            last_index += X.shape[0]

        assert last_index == X_whole.shape[0]

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_normalization_should_not_include_validation_data(self, mock_read):
        ts_size = 2
        segment_size = 4
        ts_window = 1
        validation_fraction = 0
        debug_mode = True
        fe = FeatureExtraction(ts_size, validation_fraction, segment_size=segment_size)
        X_df, y_df = fe.get_X_y(test_df)
        dt = Data(
            ts_window,
            ts_size,
            '',
            validation_fraction=0,
            segment_size=segment_size,
            normalize=False,
        )

        # We computed the scale from first 2 entries, entries which comprise the training data.
        # this is what we want to ensure in this test.
        scale = X_df.iloc[:2].abs().max()
        # get X and y from whole data.
        X_whole, y_whole = dt.get_window_X_y(X_df / scale, y_df)

        validation_fraction = 0.8
        dt = Data(
            ts_window,
            ts_size,
            '',
            validation_fraction=validation_fraction,
            segment_size=segment_size,
            normalize=True,
            ls_batch_size=segment_size,
        )

        # now X and y are fetched from generator and we ensure that it matches completely with above data.
        gen = dt.get_X_y_generator(debug_mode=debug_mode)
        first_index = 0

        for X, y in gen:
            assert (np.nan_to_num(X) == np.nan_to_num(X_whole[first_index:first_index + X.shape[0]])).all()
            assert (y == y_whole[first_index:first_index + y.shape[0]]).all()
            first_index += X.shape[0]

        # Only first segment in training data.
        assert first_index == 2

        val_X, val_y = dt.get_validation_X_y()
        assert val_X.shape[0] == 4
        assert (np.nan_to_num(val_X) == np.nan_to_num(X_whole[2:])).all()
        assert (val_y == y_whole[2:]).all()
