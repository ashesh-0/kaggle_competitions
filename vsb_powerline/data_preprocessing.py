from multiprocessing import Pool
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import numpy as np


class DataPipeline:
    """
    Top level class which takes in the parquet file and converts it to a low dimensional dataframe.
    """

    def __init__(
            self,
            parquet_fname: str,
            data_processor_args,
            start_row_num: int,
            num_rows: int,
            concurrency: int = 100,
    ):
        self._fname = parquet_fname
        self._concurrency = concurrency
        self._nrows = num_rows
        self._start_row_num = start_row_num
        self._processor_args = data_processor_args
        self._process_count = 4

    @staticmethod
    def run_one_chunk(arg_tuple):
        fname, processor_args, start_row, end_row, num_rows = arg_tuple
        cols = [str(i) for i in range(start_row, end_row)]
        data = pq.read_pandas(fname, columns=cols)
        data_df = data.to_pandas()
        processor = DataProcessor(**processor_args)
        output_df = processor.transform(data_df)
        print('Another ', round((end_row - start_row) / num_rows * 100, 2), '% Complete')
        del processor
        del data_df
        del data
        return output_df

    def run(self):
        outputs = []

        args_list = []
        for s_index in range(self._start_row_num, self._start_row_num + self._nrows, self._concurrency):
            e_index = s_index + self._concurrency
            args_this_chunk = [self._fname, self._processor_args, s_index, e_index, self._nrows]
            args_list.append(args_this_chunk)

        pool = Pool(self._process_count)
        outputs = pool.map(DataPipeline.run_one_chunk, args_list)
        pool.close()
        pool.join()

        final_output_df = pd.concat(outputs, axis=1)
        return final_output_df


class DataProcessor:
    def __init__(
            self,
            intended_time_steps: int,
            original_time_steps: int,
            peak_threshold: int,
            smoothing_window: int = 10,
            num_processes: int = 7,
    ):
        self._o_steps = original_time_steps
        self._steps = intended_time_steps
        # 50 is a confident smoothed version
        # resulting signal after subtracting the smoothened version has some noise along with signal.
        # with 10, smoothened version seems to have some signals as well.
        self._smoothing_window = smoothing_window
        self._num_processes = num_processes
        self._peak_threshold = peak_threshold

    def get_noise(self, X_df: pd.DataFrame):
        """
        TODO: we need to keep the noise. However, we don't want very high freq jitter.
        band pass filter is what is needed.
        """
        msg = 'Expected len:{}, found:{}'.format(self._o_steps, len(X_df))
        assert self._o_steps == X_df.shape[0], msg
        smoothe_df = X_df.rolling(self._smoothing_window, min_periods=1).mean()
        noise_df = X_df - smoothe_df
        del smoothe_df
        return noise_df

    @staticmethod
    def peak_stats(ser: pd.Series, threshold, quantiles=[0, 0.25, 0.5, 0.75, 1]):
        """
        Returns quantiles of peak width, height, distances from next peak.
        """
        maxima_peak_indices, maxima_data_dict = find_peaks(ser, threshold=threshold, width=1)
        maxima_width = maxima_data_dict['widths']
        maxima_height = maxima_data_dict['prominences']

        minima_peak_indices, minima_data_dict = find_peaks(-1 * ser, threshold=threshold, width=1)
        minima_width = minima_data_dict['widths']
        minima_height = minima_data_dict['prominences']

        peak_indices = np.concatenate([maxima_peak_indices, minima_peak_indices])
        peak_width = np.concatenate([maxima_width, minima_width])
        peak_height = np.concatenate([maxima_height, minima_height])
        maxima_minima = np.concatenate([np.array([1] * len(maxima_height)), np.array([-1] * len(minima_height))])

        index_ordering = np.argsort(peak_indices)
        peak_width = peak_width[index_ordering]
        peak_height = peak_height[index_ordering]
        peak_indices = peak_indices[index_ordering]
        maxima_minima = maxima_minima[index_ordering]

        if len(peak_indices) == 0:
            # no peaks
            width_stats = [0] * len(quantiles)
            height_stats = [0] * len(quantiles)
            distance_stats = [0] * len(quantiles)
        else:
            peak_distances = np.diff(peak_indices)

            peak_width[peak_width > 100] = 100

            width_stats = np.quantile(peak_width, quantiles)
            height_stats = np.quantile(peak_height, quantiles)

            # for just one peak, distance will be empty array.
            if len(peak_distances) == 0:
                assert len(peak_indices) == 1
                distance_stats = [ser.shape[0]] * len(quantiles)
            else:
                distance_stats = np.quantile(peak_distances, quantiles)

        width_names = ['peak_width_' + str(i) for i in quantiles]
        height_names = ['peak_height_' + str(i) for i in quantiles]
        distance_names = ['peak_distances_' + str(i) for i in quantiles]

        index = width_names + height_names + distance_names + ['peak_count']
        data = np.concatenate([width_stats, height_stats, distance_stats, [len(peak_indices)]])

        return pd.Series(data, index=index)

    @staticmethod
    def get_peak_stats_df(df, peak_threshold):
        """
        Args:
            df:
                columns are different examples.
                axis is time series.
        """
        return df.apply(lambda x: DataProcessor.peak_stats(x, peak_threshold), axis=0)

    @staticmethod
    def pandas_describe(df):
        output_df = df.quantile([0, 0.25, 0.5, 0.75, 1], axis=0)
        output_df.index = list(map(lambda x: 'Quant-' + str(x), output_df.index.tolist()))
        abs_mean_df = df.abs().mean().to_frame('abs_mean')
        mean_df = df.mean().to_frame('mean')
        std_df = df.std().to_frame('std')
        return pd.concat([output_df, abs_mean_df.T, mean_df.T, std_df.T])

    @staticmethod
    def transform_chunk(signal_time_series_df: pd.DataFrame, peak_threshold: float) -> pd.DataFrame:
        """
        It sqashes the time series to a single point multi featured vector.
        """
        df = signal_time_series_df
        # mean, var, percentile.
        # NOTE pandas.describe() is the costliest computation with 95% time of the function.
        metrics_df = DataProcessor.pandas_describe(df)
        peak_metrics_df = DataProcessor.get_peak_stats_df(df, peak_threshold)

        metrics_df.index = list(map(lambda x: 'signal_' + x, metrics_df.index))
        temp_metrics = [metrics_df, peak_metrics_df]

        for smoothener in [1, 2, 4, 8, 16, 32]:
            diff_df = df.rolling(smoothener).mean()[::smoothener].diff().abs()
            temp_df = DataProcessor.pandas_describe(diff_df)

            temp_df.index = list(map(lambda x: 'diff_smoothend_by_' + str(smoothener) + ' ' + x, temp_df.index))
            temp_metrics.append(temp_df)

        df = pd.concat(temp_metrics)
        df.index.name = 'features'
        # In total there are 6*
        return df

    def transform(self, X_df: pd.DataFrame):
        """
        Args:
            X_df: dataframe with each column being one data point. Rows are timestamps.
        """
        # Remove the smoothened version of the data so as to work with noise.
        if self._smoothing_window > 0:
            X_df = self.get_noise(X_df)

        # stepsize many consequitive timestamps are compressed to form one timestamp.
        # this will ensure we are left with self._steps many timestamps.
        stepsize = self._o_steps // self._steps
        transformed_data = []
        for s_tm_index in range(0, self._o_steps, stepsize):
            e_tm_index = s_tm_index + stepsize
            # NOTE: dask was leading to memory leak.
            #one_data_point = delayed(DataProcessor.transform_chunk)(X_df.iloc[s_tm_index:e_tm_index, :])
            one_data_point = DataProcessor.transform_chunk(X_df.iloc[s_tm_index:e_tm_index, :], self._peak_threshold)
            transformed_data.append(one_data_point)

        # transformed_data = dd.compute(*transformed_data, scheduler='processes', num_workers=self._num_processes)
        # Add timestamp
        for ts in range(0, len(transformed_data)):
            transformed_data[ts]['ts'] = ts

        df = pd.concat(transformed_data, axis=0).set_index(['ts'], append=True)
        df.columns.name = 'Examples'
        return df
