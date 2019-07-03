import numpy as np
import pandas as pd


def get_structure_data(X, str_df):
    """
    Adds atom and xyz for both nodes of the bond 'atom_index_0','atom_index_1'
    """
    X = X.reset_index()
    X = pd.merge(
        X, str_df, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
    X.drop(['atom_index'], inplace=True, axis=1)
    X.rename({'x': 'x_0', 'y': 'y_0', 'z': 'z_0', 'atom': 'atom_0'}, axis=1, inplace=True)

    X = pd.merge(
        X, str_df, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left')
    X.drop(['atom_index'], inplace=True, axis=1)
    X.rename({'x': 'x_1', 'y': 'y_1', 'z': 'z_1', 'atom': 'atom_1'}, axis=1, inplace=True)
    X.set_index('id',inplace=True)
    return X


def find_distance_from_plane(df, x: str, y: str, z: str):
    return (df[x] * df['m_x'] + df[y] * df['m_y'] + df[z] * df['m_z'] + df['c']) / df['m_2norm']


def find_distance_btw_point(df, x0, y0, z0, x1, y1, z1):
    return np.sqrt((df[x0] - df[x1]).pow(2) + (df[y0] - df[y1]).pow(2) + (df[z0] - df[z1]).pow(2))


def find_cos(df, x_A, y_A, z_A, x_B, y_B, z_B, x_C, y_C, z_C):
    """
    Finds cosine of angle ABC
    """
    A = df[[x_A, y_A, z_A]]
    B = df[[x_B, y_B, z_B]]
    C = df[[x_C, y_C, z_C]]
    A.columns = [0, 1, 2]
    B.columns = [0, 1, 2]
    C.columns = [0, 1, 2]
    vector1 = A - B
    vector1 = vector1.divide(np.linalg.norm(vector1, axis=1), axis=0)
    vector2 = C - B
    vector2 = vector2.divide(np.linalg.norm(vector2, axis=1), axis=0)
    return (vector1 * vector2).sum(axis=1)
