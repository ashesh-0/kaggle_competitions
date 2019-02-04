from typing import List

import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# class DataPipeline:
#     """
#     Top level class which takes in the parquet file and converts it to a low dimensional dataframe.
#     """

#     def __init__(
#             self,
#             parquet_fname: str,
#             data_processor,
#             concurrency: int = 10,
#             num_rows: int = 1782,
#     ):
#         self._fname = parquet_fname
#         self._concurrency = concurrency
#         self._nrows = num_rows
#         self._processor = data_processor

#     def run(self):
#         outputs = []
#         for s_index in range(0, self._nrows, self._concurrency):
#             e_index = s_index + self._concurrency
#             cols = [str(i) for i in range(s_index, e_index)]
#             data_df = pq.read_pandas(self._fname, columns=cols).to_pandas()
#             output_df = self._processor.transform(data_df)
#             outputs.append(output_df)
#             print('[Data Processing]', round(e_index / self._nrows * 100, 1), '% complete')


class DataProcessor:
    def __init__(self, intended_time_steps: int, original_time_steps: int, smoothing_window: int = 10):
        self._o_steps = original_time_steps
        self._steps = intended_time_steps
        # 50 is a confident smoothed version
        # resulting signal after subtracting the smoothened version has some noise along with signal.
        # with 10, smoothened version seems to have some signals as well.
        self._smoothing_window = smoothing_window

    def get_noise(self, X_df: pd.DataFrame):
        """
        TODO: we need to keep the noise. However, we don't want very high freq jitter.
        band pass filter is what is needed.
        """
        assert self._o_steps == X_df.shape[0], 'Expected len:{}, found:{}'.format(self._o_steps, len(X))
        noise_df = X_df - X_df.rolling(self._smoothing_window, min_periods=1).mean()
        return noise_df

    def transform_chunk(self, signal_time_series_df: list) -> np.array:
        """
        It sqashes the time series to a single point multi featured vector.
        """
        df = signal_time_series_df
        # mean, var, percentile.
        metrics_df = df.describe()

        metrics_df.index = list(map(lambda x: 'signal_' + x, metrics_df.index))
        temp_metrics = [metrics_df]

        for smoothener in [1, 2, 4, 8, 16, 32]:
            diff_df = df.rolling(smoothener).mean()[::smoothener].diff()
            temp_df = diff_df.describe()

            temp_df.index = list(map(lambda x: 'diff_smoothend_by_' + str(smoothener) + ' ' + x, temp_df.index))
            temp_metrics.append(temp_df)

        return pd.concat(temp_metrics)

    def transform(self, X_df: List[float]):
        # Work with noise
        if self._smoothing_window > 0:
            X_df = self.get_noise(X_df)

        stepsize = self._o_steps // self._steps
        transformed_data = []
        i = 0
        for s_tm_index in range(0, self._o_steps, stepsize):
            e_tm_index = s_tm_index + stepsize
            one_data_point = self.transform_chunk(X_df.iloc[s_tm_index:e_tm_index, :])
            one_data_point['ts'] = i
            transformed_data.append(one_data_point)
            i += 1

        return pd.concat(transformed_data, axis=0).set_index(['ts'], append=True)
