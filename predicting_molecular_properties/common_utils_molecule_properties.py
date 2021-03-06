import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def get_edges_df_from_obabel(obabel_edges_df, structures_df):
    """
    Given obabel edge data, convert it to format of edges_df so that all other code remains the same.
    """
    obabel_edges_df = obabel_edges_df[[
        'mol_id', 'atom_index_0', 'atom_index_1', 'BondLength', 'BondOrder', 'IsAromatic'
    ]]
    obabel_edges_df.loc[obabel_edges_df.IsAromatic, 'BondOrder'] = 1.5
    obabel_edges_df.rename({'BondLength': 'distance', 'BondOrder': 'bond_type'}, axis=1, inplace=True)
    enc = LabelEncoder()
    enc.fit(structures_df.molecule_name)
    obabel_edges_df['molecule_name'] = enc.inverse_transform(obabel_edges_df['mol_id'])
    obabel_edges_df.drop(['IsAromatic', 'mol_id'], axis=1, inplace=True)

    obabel_edges_df = add_structure_data_to_edge(obabel_edges_df, structures_df, ['x', 'y', 'z'])
    obabel_edges_df['x'] = (obabel_edges_df['x_1'] - obabel_edges_df['x_0']) / obabel_edges_df['distance']
    obabel_edges_df['y'] = (obabel_edges_df['y_1'] - obabel_edges_df['y_0']) / obabel_edges_df['distance']
    obabel_edges_df['z'] = (obabel_edges_df['z_1'] - obabel_edges_df['z_0']) / obabel_edges_df['distance']
    obabel_edges_df.drop(['x_1', 'x_0', 'y_1', 'y_0', 'z_0', 'z_1'], axis=1, inplace=True)
    return obabel_edges_df


def add_structure_data_to_edge(edges_df, structures_df, cols_to_add=['atom']):
    edges_df = pd.merge(
        edges_df,
        structures_df[['molecule_name', 'atom_index'] + cols_to_add],
        how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'],
    )
    edges_df.drop('atom_index', axis=1, inplace=True)

    edges_df.rename({k: f'{k}_0' for k in cols_to_add}, inplace=True, axis=1)

    edges_df = pd.merge(
        edges_df,
        structures_df[['molecule_name', 'atom_index'] + cols_to_add],
        how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'],
    )
    edges_df.drop('atom_index', axis=1, inplace=True)
    edges_df.rename({k: f'{k}_1' for k in cols_to_add}, inplace=True, axis=1)
    return edges_df


def get_symmetric_edges(edge_df):
    """
    Ensures that all edges in all molecules occur exactly twice in edge_df. This ensures that when we join with
    on with one of atom_index_0/atom_index_1, all edges are covered.
    """
    e_df = edge_df.copy()
    atom_1 = e_df.atom_index_1.copy()
    e_df['atom_index_1'] = e_df['atom_index_0']
    e_df['atom_index_0'] = atom_1
    xyz_cols = list(set(['x', 'y', 'z']).intersection(set(edge_df.columns)))
    assert len(xyz_cols) in [0, 3]
    if len(xyz_cols) == 3:
        e_df[['x', 'y', 'z']] = -1 * e_df[['x', 'y', 'z']]

    edge_df = pd.concat([edge_df, e_df], ignore_index=True)
    return edge_df


def get_structure_data(X, str_df, cols_to_add=None):
    """
    Adds atom and xyz for both nodes of the bond 'atom_index_0','atom_index_1'
    """
    if cols_to_add is None:
        cols_to_add = ['x', 'y', 'z', 'atom']

    str_df = str_df[cols_to_add + ['molecule_name', 'atom_index']]

    X = X.reset_index()
    X = pd.merge(
        X, str_df, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
    X.drop(['atom_index'], inplace=True, axis=1)
    X.rename({c: f'{c}_0' for c in cols_to_add}, axis=1, inplace=True)

    X = pd.merge(
        X, str_df, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left')
    X.drop(['atom_index'], inplace=True, axis=1)
    X.rename({c: f'{c}_1' for c in cols_to_add}, axis=1, inplace=True)
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


def find_perpendicular_to_plane(df, df_plane, x, y, z, mx, my, mz):
    assert df.index.equals(df_plane.index)
    m_norm = np.sqrt(np.square(df_plane[[mx, my, mz]]).sum(axis=1))
    output_df = pd.Series(
        dot(df, df_plane, [x, y, z], [mx, my, mz]) / m_norm, index=df.index).to_frame('plane_component')
    output_df['x'] = (df_plane[mx] / m_norm) * output_df['plane_component']
    output_df['y'] = (df_plane[my] / m_norm) * output_df['plane_component']
    output_df['z'] = (df_plane[mz] / m_norm) * output_df['plane_component']
    return output_df[['x', 'y', 'z']]


def find_projection_on_plane(df, df_plane, x, y, z, mx, my, mz, normalize=True):
    output_df = find_perpendicular_to_plane(df, df_plane, x, y, z, mx, my, mz)
    assert set(output_df.columns) == set(['x', 'y', 'z'])
    output_df['x'] = df[x] - output_df['x']
    output_df['y'] = df[y] - output_df['y']
    output_df['z'] = df[z] - output_df['z']
    if normalize:
        length = np.sqrt(np.square(output_df).sum(axis=1))
        output_df = output_df.divide(length, axis=0)
    return output_df


def find_cross_product(df, x_A, y_A, z_A, x_B, y_B, z_B, normalize=True):
    """
    Cross product of two vectors
    """
    cross_df = pd.DataFrame([], index=df.index)
    cross_df['x'] = df[y_A] * df[z_B] - df[y_B] * df[z_A]
    cross_df['y'] = df[z_A] * df[x_B] - df[z_B] * df[x_A]
    cross_df['z'] = df[x_A] * df[y_B] - df[x_B] * df[y_A]
    if normalize:
        d = np.sqrt(np.square(cross_df).sum(axis=1))
        cross_df = cross_df.divide(d, axis=0)

    return cross_df


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
