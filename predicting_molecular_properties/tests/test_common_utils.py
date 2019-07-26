import numpy as np
import pandas as pd
from common_utils_molecule_properties import (find_cos, find_cross_product, find_perpendicular_to_plane,
                                              find_projection_on_plane)


def test_find_projection_on_plane():
    df = pd.DataFrame(
        [
            [2, 1, 1, 2, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        columns=['x1', 'y1', 'z1', 'm1x', 'm1y', 'm1z'])
    output_df = find_projection_on_plane(
        df[['x1', 'y1', 'z1']], df[['m1x', 'm1y', 'm1z']], 'x1', 'y1', 'z1', 'm1x', 'm1y', 'm1z', normalize=False)
    expected_output_df = pd.DataFrame(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 0],
        ], columns=['x', 'y', 'z'])
    assert ((output_df - expected_output_df).abs().sum().sum() < 1e-14)


def test_find_perpendicular_to_plane():
    df = pd.DataFrame(
        [
            [2, 1, 1, 2, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        columns=['x', 'y', 'z', 'mx', 'my', 'mz'])
    output_df = find_perpendicular_to_plane(df, df, 'x', 'y', 'z', 'mx', 'my', 'mz')
    expected_output_df = pd.DataFrame(
        [
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], columns=['x', 'y', 'z'])
    assert ((output_df - expected_output_df).abs().sum().sum() < 1e-14)


def test_cross_product():
    df = pd.DataFrame(
        [
            [1, 0, 0, 0, 1.0, 0],
            [1, 0, 0, 0, 0.1, 0],
            [0, 1, 0, 0, 0.0, 1],
            [0, 0, 1, 0, 1.0, 0],
            [1, 0, 0, 0, 0.0, 1],
            [1, 1, 1, 0, 0.0, 1],
            [1, 2, 3, 1, 2.0, 3],
        ],
        columns=['x0', 'y0', 'z0', 'x1', 'y1', 'z1'])
    output_df = find_cross_product(df, 'x0', 'y0', 'z0', 'x1', 'y1', 'z1')
    expected_output_df = pd.DataFrame(
        [
            [0, 0, 1.],
            [0, 0, 1.],
            [1, 0, 0.],
            [-1, 0, 0.],
            [0, -1, 0.],
            [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            [0, 0, 0],
        ],
        columns=['x', 'y', 'z'])

    assert ((output_df - expected_output_df).abs().sum().sum() < 1e-14)


def test_find_cos():
    df = pd.DataFrame([[0, 0, 0, 1, 0, 0, 1, 1, 0]])
    df.loc[1] = [0, 0, 0, 1, 2, 0, -1, 3, 0]
    df.loc[2] = [0, 0, 0, 1, 2, 0, 2, 4, 0]
    cos = find_cos(df, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert cos.loc[0] == 0
    assert cos.loc[1] == 0
    assert abs(cos.loc[2] + 1) < 1e-6
