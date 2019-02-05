from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq


class DataPipeline:
    """
    Top level class which takes in the parquet file and converts it to a low dimensional dataframe.
    """

    def __init__(
            self,
            parquet_fname: str,
            data_processor_args,
            concurrency: int = 100,
            num_rows: int = 8712,
    ):
        self._fname = parquet_fname
        self._concurrency = concurrency
        self._nrows = num_rows
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
        print(round(end_row / num_rows * 100, 2), '% Complete')
        del processor
        del data_df
        del data
        return output_df

    def run(self):
        outputs = []

        args_list = []
        for s_index in range(0, self._nrows, self._concurrency):
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

    def get_noise(self, X_df: pd.DataFrame):
        """
        TODO: we need to keep the noise. However, we don't want very high freq jitter.
        band pass filter is what is needed.
        """
        assert self._o_steps == X_df.shape[0], 'Expected len:{}, found:{}'.format(self._o_steps, len(X))
        smoothe_df = X_df.rolling(self._smoothing_window, min_periods=1).mean()
        noise_df = X_df - smoothe_df
        del smoothe_df
        return noise_df

    @staticmethod
    def pandas_describe(df):
        output_df = df.quantile([0, 0.25, 0.5, 0.75, 1], axis=0)
        output_df.index = list(map(lambda x: 'Quant-' + str(x), output_df.index.tolist()))
        abs_mean_df = df.abs().mean().to_frame('abs_mean')
        mean_df = df.mean().to_frame('mean')
        std_df = df.std().to_frame('std')
        return pd.concat([output_df, abs_mean_df.T, mean_df.T, std_df.T])

    @staticmethod
    def transform_chunk(signal_time_series_df: pd.DataFrame) -> pd.DataFrame:
        """
        It sqashes the time series to a single point multi featured vector.
        """
        df = signal_time_series_df
        # mean, var, percentile.
        # NOTE pandas.describe() is the costliest computation with 95% time of the function.
        metrics_df = DataProcessor.pandas_describe(df)

        metrics_df.index = list(map(lambda x: 'signal_' + x, metrics_df.index))
        temp_metrics = [metrics_df]

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
            one_data_point = DataProcessor.transform_chunk(X_df.iloc[s_tm_index:e_tm_index, :])
            transformed_data.append(one_data_point)

        # transformed_data = dd.compute(*transformed_data, scheduler='processes', num_workers=self._num_processes)
        # Add timestamp
        for ts in range(0, len(transformed_data)):
            transformed_data[ts]['ts'] = ts

        df = pd.concat(transformed_data, axis=0).set_index(['ts'], append=True)
        df.columns.name = 'Examples'
        return df
