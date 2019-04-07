from unittest.mock import patch
import pandas as pd
from data import FeatureExtraction, Data

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
    return test_df.copy()


class TestFeatureExtraction:
    def test_get_X_y(self):
        ts_size = 3
        ex = FeatureExtraction(ts_size)
        X_df, y_df = ex.get_X_y(test_df)

        assert X_df.shape[0] == y_df.shape[0]
        assert y_df.shape[0] == test_df.shape[0] // 3

        expected_y = test_df['time_to_failure'].shift(-1 * ts_size + 1)[::ts_size]
        expected_y.index = list(range(test_df.shape[0] // ts_size))
        assert y_df.equals(expected_y)

        assert X_df['mean'].iloc[0] == test_df['acoustic_data'].iloc[:ts_size].mean()
        assert X_df['mean'].iloc[1] == test_df['acoustic_data'].iloc[ts_size:2 * ts_size].mean()
        assert X_df['mean'].iloc[2] == test_df['acoustic_data'].iloc[2 * ts_size:3 * ts_size].mean()
        assert X_df['mean'].iloc[3] == test_df['acoustic_data'].iloc[3 * ts_size:4 * ts_size].mean()

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_X_y_generator(self, mock_r):
        ts_size = 2
        segment_size = 4
        padding = 0
        ex = FeatureExtraction(ts_size, segment_size=segment_size)
        gen = ex.get_X_y_generator('', padding)

        expected_y = test_df['time_to_failure'].shift(-1 * ts_size + 1)[::ts_size]
        expected_y.index = list(range(test_df.shape[0] // ts_size))
        num_ts_in_segment = segment_size // ts_size

        for i, data in enumerate(gen):
            X_df, y_df = data
            assert y_df.shape[0] == X_df.shape[0] == segment_size / ts_size
            assert expected_y[i * num_ts_in_segment:(i + 1) * num_ts_in_segment].equals(y_df)

        assert (i + 1) == test_df.shape[0] / segment_size

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_X_y_generator_with_padding(self, mock_r):
        ts_size = 2
        segment_size = 6
        padding = 2
        ex = FeatureExtraction(ts_size, segment_size=segment_size)
        gen = ex.get_X_y_generator('', padding * ts_size)

        expected_y = test_df['time_to_failure'].shift(-1 * ts_size + 1)[::ts_size]
        expected_y.index = list(range(test_df.shape[0] // ts_size))
        num_ts_in_segment = segment_size // ts_size

        for i, data in enumerate(gen):
            X_df, y_df = data
            # In first segment, no way to have padding
            if i == 0:
                assert y_df.shape[0] == X_df.shape[0] == (segment_size / ts_size)
            else:
                assert y_df.shape[0] == X_df.shape[0] == (segment_size / ts_size + padding)

            assert expected_y[max(0, i * num_ts_in_segment - padding):(i + 1) * num_ts_in_segment].equals(y_df)

        assert (i + 1) == test_df.shape[0] / segment_size


class TestData:
    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_window_X_y(self, mock_):
        ts_size = 2
        segment_size = 6
        ts_window = 3
        fe = FeatureExtraction(ts_size, segment_size=segment_size)
        X_df, y_df = fe.get_X_y(test_df)

        dt = Data(ts_window, ts_size)
        X, y = dt.get_window_X_y(X_df, y_df)
        assert X.shape == (X_df.shape[0] - ts_window + 1, ts_window, X_df.shape[1])
        assert y.shape == (X_df.shape[0] - ts_window + 1, )

        for i in range(X.shape[0]):
            assert y_df.values[i + ts_window - 1] == y[i]
            for j in range(ts_window):
                assert all(X[i, j, :] == X_df.values[i + j, :])

    @patch('data.pd.read_csv', side_effect=mock_read_csv)
    def test_get_X_y_generator(self, mock_):
        """
        Tests that whether one computes X,y in one go or computes through generator, resulting data is the same.
        """
        ts_size = 2
        segment_size = 6
        ts_window = 3
        fe = FeatureExtraction(ts_size, segment_size=segment_size)
        X_df, y_df = fe.get_X_y(test_df)

        dt = Data(ts_window, ts_size)
        # get X and y from whole data.
        X_whole, y_whole = dt.get_window_X_y(X_df, y_df)

        # now X and y are fetched from generator and we ensure that it matches completely with above data.
        gen = dt.get_X_y_generator('')
        last_index = 0
        for X, y in gen:
            assert (X == X_whole[last_index:last_index + X.shape[0]]).all()
            assert (y == y_whole[last_index:last_index + y.shape[0]]).all()
            last_index += X.shape[0]

        assert last_index == X_whole.shape[0]
