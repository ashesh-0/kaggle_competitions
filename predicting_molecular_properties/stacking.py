import numpy as np
import pandas as pd


def find_best_linear_combination(df1, df2, actual, metric_fn, start, end, stepsize):
    """
    df1 and df2 have to be pd.Series.
    actual can be a df/series. it will be used only in metric_fn:
            metric_fn(actual,prediction_df)

    """
    assert start >= 0
    assert end <= 1

    best_score = None
    best_alpha = None

    alphas = list(np.arange(start, end, stepsize))
    if end not in alphas:
        alphas.append(end)

    for alpha in alphas:
        combined = alpha * df1 + (1 - alpha) * df2
        score = metric_fn(actual, combined)
        # print(alpha, score)
        if best_score is None or score < best_score:
            best_score = score
            best_alpha = alpha

    # import pdb
    # pdb.set_trace()
    return (best_score, best_alpha)


class LinearStacking:
    def __init__(self, metric_fn, starting_alpha=0, ending_alpha=1, stepsize=0.01):
        self._metric_fn = metric_fn
        self._start = starting_alpha
        self._end = ending_alpha
        self._stepsize = stepsize

        self._weights = []

    def check_weights_sanity(self):
        if len(self._weights) == 0:
            return
        if len(self._weights) == 1:
            assert self._weights[0] <= 1
            return

        assert all([w <= 1 and w >= 0 for w in self._weights])

        weights = self._weights.copy()
        weights = [0] + weights
        total_weights = []
        for i in range(len(weights)):
            ith_df_weight = (1 - weights[i]) * np.product(weights[i + 1:])
            total_weights.append(ith_df_weight)

        print('Weights: ', ' '.join(['{:3f}'.format(wi) for wi in total_weights]))
        assert abs(np.sum(total_weights) - 1) < 1e-10

    def fit(self, df_array, target_y):
        if len(df_array) == 1:
            print('Just one model. Allocating everyting to it.')
            return

        cumulative_best_df = df_array[0]
        self._weights = []
        for i in range(1, len(df_array)):
            best_score, best_alpha = find_best_linear_combination(
                cumulative_best_df,
                df_array[i],
                target_y,
                self._metric_fn,
                self._start,
                self._end,
                self._stepsize,
            )
            cumulative_best_df = cumulative_best_df * best_alpha + df_array[i] * (1 - best_alpha)
            self._weights.append(best_alpha)
            print('{}th inclusion BestScore:{:.3f}'.format(i, best_score))

        print('Individual best performance:',
              ' '.join(['{:3f}'.format(self._metric_fn(target_y, df)) for df in df_array]))
        print('Median best performance: {:3f}'.format(
            self._metric_fn(target_y, pd.concat(df_array, axis=1).median(axis=1))))
        self.check_weights_sanity()

    def transform(self, df_array):
        assert len(df_array) >= 1
        assert len(df_array) == len(self._weights) + 1
        if len(df_array) == 1:
            return df_array[0]

        cumulative_best_df = df_array[0]
        for i in range(1, len(df_array)):
            alpha = self._weights[i - 1]
            cumulative_best_df = alpha * cumulative_best_df + (1 - alpha) * df_array[i]

        return cumulative_best_df
