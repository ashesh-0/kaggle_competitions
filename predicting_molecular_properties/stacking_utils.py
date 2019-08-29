import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


def _get_distance_factor(row):
    num_models = len(row)
    A = np.tile(row, (num_models, 1)).T
    dist_matrix = np.abs(A - A.T)
    avg_distance_to_rest_models = np.sum(
        dist_matrix, axis=1) / (num_models - 1)

    avg_distance_btw_rest_models = (
        np.sum(dist_matrix) - 2 * np.sum(dist_matrix, axis=1)) / (
            (num_models - 1) * (num_models - 2))

    factor = avg_distance_to_rest_models / avg_distance_btw_rest_models
    return np.where(avg_distance_to_rest_models > 5, factor, 1)


def skip_outlier(df, weights, outlier_threshold):
    assert np.sum(weights) == 1
    assert np.sum(np.abs(weights)) == 1
    assert df.shape[1] == len(weights)

    outliers_skipped = 0

    weights = np.array(weights)
    data = df.values
    prediction = np.zeros((df.shape[0], 1))
    for index, row in enumerate(tqdm_notebook(data)):
        distance_factor = _get_distance_factor(row)
        mask = (distance_factor < outlier_threshold)

        # atmost one can be skipped. There is a possibility of 2 getting skipped if 2 are on either side.
        assert np.sum(mask) >= df.shape[1] - 1
        outliers_skipped += df.shape[1] - np.sum(mask)
        new_weights = weights[mask]
        new_weights = new_weights / np.sum(new_weights)
        prediction[index, 0] = np.sum(row[mask] * new_weights)

    print('Outliers skipped', 100 * outliers_skipped / df.shape[0], '%')

    return pd.DataFrame(prediction, index=df.index, columns=['prediction'])




if __name__ == '__main__':
    print(_get_distance_factor([2, 1, 3, 2]))
