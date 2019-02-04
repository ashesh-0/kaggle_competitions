import scipy.fftpack as fftpack

import pyarrow.parquet as pq
import pandas as pd
import numpy as np


class DataPipeline:
    """
    Top level class which takes in the parquet file and converts it to a low dimensional dataframe.
    """

    def __init__(
            self,
            parquet_fname: str,
            data_processor,
            concurrency: int = 10,
            num_rows: int = 1782,
    ):
        self._fname = parquet_fname
        self._concurrency = concurrency
        self._nrows = num_rows
        self._processor = data_processor

    def run(self):
        outputs = []
        for s_index in range(0, self._nrows, self._concurrency):
            e_index = s_index + self._concurrency
            cols = [str(i) for i in range(s_index, e_index)]
            data_df = pq.read_pandas(self._fname, columns=cols).to_pandas()
            output_df = self._processor.transform(data_df)
            outputs.append(output_df)
            print('[Data Processing]', round(e_index / self._nrows * 100, 1), '% complete')

        return pd.concat(outputs)


class DataProcessor:
    def __init__(self, intended_time_steps: int, original_time_steps: int, keep_frequencies: int = -1):
        self._o_steps = original_time_steps
        self._steps = intended_time_steps
        self._freq = keep_frequencies

    def fourier_denoise(self, X: list[int]):
        """
        TODO: we need to keep the noise. However, we don't want very high freq jitter.
        band pass filter is what is needed.
        """
        return X
        # return fftpack.irfft(fftpack.rfft(X)[:self._freq])

    def transform_chunk(self, signal_time_series: list) -> np.array:
        """
        It sqashes the time series to a single point multi featured vector.
        """
        df = pd.DataFrame(signal_time_series.reshape(len(signal_time_series), 1), columns=['signal'])
        # mean, var, percentile.
        metrics = df['signal'].describe().values

        for smoothener in [1, 2, 4, 8, 16, 32]:
            metrics += df['signal'].rolling(smoothener)[::smoothener].diff().describe().values

        return metrics.reshape(1, len(metrics))

    def transform(self, X):
        # Fourier Denoising
        if self._freq > 0:
            X = self.fourier_denoise(X)

        stepsize = self._o_steps // self._steps
        for s_tm_index in range(0, self._o_steps, stepsize):
            e_tm_index = s_tm_index + stepsize

            X[s_tm_index:e_tm_index]
