from common_utils import Scaler
import pandas as pd
import numpy as np


def test_scaler_should_normalize_all_columns():
    df = pd.DataFrame(np.random.rand(10, 5))
    df_orig = df.copy()

    std_df = df.std(ddof=0)
    mean_df = df.mean()

    sc = Scaler()
    sc.fit_transform(df)
    assert (df.mean() - 0).abs().max() < 1e-14
    assert (df.std(ddof=0) - 1).abs().max() < 1e-14
    assert ((df_orig - mean_df) / std_df - df).abs().max().max() < 1e-14


def test_scaler_should_skip_given_columns():
    df = pd.DataFrame(np.random.rand(10, 5))
    df_orig = df.copy()

    std_df = df.std(ddof=0)
    mean_df = df.mean()

    unchanged_columns = [1, 4]
    sc = Scaler(skip_columns=unchanged_columns)
    sc.fit_transform(df)
    normalized_columns = [0, 2, 3]
    assert (df.mean().loc[normalized_columns] - 0).abs().max() < 1e-14
    assert (df.std(ddof=0).loc[normalized_columns] - 1).abs().max() < 1e-14
    assert ((df_orig - mean_df) / std_df - df).abs()[normalized_columns].max().max() < 1e-14
    assert (df[unchanged_columns] - df_orig[unchanged_columns]).abs().max().max() < 1e-14
