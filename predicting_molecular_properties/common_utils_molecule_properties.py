import numpy as np
import pandas as pd


def add_structure_data_to_edge(edges_df, structures_df):
    edges_df = pd.merge(
        edges_df,
        structures_df[['molecule_name', 'atom_index', 'atom']],
        how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'],
    )
    edges_df.drop('atom_index', axis=1, inplace=True)
    edges_df.rename({'atom': 'atom_0'}, inplace=True, axis=1)

    edges_df = pd.merge(
        edges_df,
        structures_df[['molecule_name', 'atom_index', 'atom']],
        how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'],
    )
    edges_df.drop('atom_index', axis=1, inplace=True)
    edges_df.rename({'atom': 'atom_1'}, inplace=True, axis=1)
    return edges_df


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
    X.set_index('id', inplace=True)
    return X


def dot(df_left, df_right, cols_left, cols_right):
    dot_df = None
    assert len(cols_left) == len(cols_right)
    assert len(set(cols_left) - set(df_left.columns.tolist())) == 0
    assert len(set(cols_right) - set(df_right.columns.tolist())) == 0

    for i in range(len(cols_left)):
        temp_sum = df_left[cols_left[i]] * df_right[cols_right[i]]
        if dot_df is None:
            dot_df = temp_sum
        else:
            dot_df += temp_sum

    return dot_df


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


def plot_molecule(structures_df, mol_name, figsize=(10, 10)):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    temp_df = structures_df[structures_df.molecule_name == mol_name]
    marker_size = {'C': 120, 'H': 10, 'N': 70, 'F': 90, 'O': 80}
    marker_color = {'C': 'blue', 'H': 'orange', 'N': 'green', 'F': 'black', 'O': 'violet'}
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    for atom in temp_df.atom.unique():
        dta = temp_df[temp_df.atom == atom]
        ax.scatter(dta.x, dta.y, dta.z, s=marker_size[atom], c=marker_color[atom], label=atom)
        for _, row in dta.iterrows():
            ax.text(row.x, row.y, row.z, row.atom_index, fontsize=20)
    ax.legend()
