import pandas as pd
import numpy as np
from decorators import timer


@timer('DistanceFeatures')
def _add_distance_from_center(X_df, structures_df):

    mean_df = structures_df.groupby('molecule_name')[['x', 'y', 'z']].mean().astype(np.float32)
    mean_df.rename({'x': 'x_mean', 'y': 'y_mean', 'z': 'z_mean'}, inplace=True, axis=1)

    X_df = X_df.join(mean_df, how='left', on='molecule_name')

    X_df['dis_x_0_mean'] = (X_df['x_0'] - X_df['x_mean']).pow(2).astype(np.float32)
    X_df['dis_y_0_mean'] = (X_df['y_0'] - X_df['y_mean']).pow(2).astype(np.float32)
    X_df['dis_z_0_mean'] = (X_df['z_0'] - X_df['z_mean']).pow(2).astype(np.float32)
    X_df['dis_0_mean'] = X_df['dis_x_0_mean'] + X_df['dis_y_0_mean'] + X_df['dis_z_0_mean']

    X_df['dis_x_1_mean'] = (X_df['x_1'] - X_df['x_mean']).pow(2).astype(np.float32)
    X_df['dis_y_1_mean'] = (X_df['y_1'] - X_df['y_mean']).pow(2).astype(np.float32)
    X_df['dis_z_1_mean'] = (X_df['z_1'] - X_df['z_mean']).pow(2).astype(np.float32)
    X_df['dis_1_mean'] = X_df['dis_x_1_mean'] + X_df['dis_y_1_mean'] + X_df['dis_z_1_mean']

    return X_df


@timer('DistanceFeatures')
def _set_distance_from_molecule_centers_feature(df):
    bond_mid_x = (df['x_0'] + df['x_1']) / 2
    bond_mid_y = (df['y_0'] + df['y_1']) / 2
    bond_mid_z = (df['z_0'] + df['z_1']) / 2
    bond_mid_dict = {'x': bond_mid_x, 'y': bond_mid_y, 'z': bond_mid_z}

    min_col = 'MolShape_{}_len_0'

    for col in ['MolShape_{}_len_1', 'MolShape_{}_len_.5', 'MolShape_{}_len_.1']:
        for dim in ['x', 'y', 'z']:
            dim_col = col.format(dim)
            dis_from_min = (bond_mid_dict[dim] - df[min_col.format(dim)])
            df['dis_' + dim_col] = (dis_from_min / df[dim_col] - 0.5).abs().fillna(100)


@timer('DistanceFeatures')
def _add_bond_distance_feature(df):
    df['dis_x'] = (df['x_0'] - df['x_1']).pow(2).astype(np.float32)
    df['dis_y'] = (df['y_0'] - df['y_1']).pow(2).astype(np.float32)
    df['dis_z'] = (df['z_0'] - df['z_1']).pow(2).astype(np.float32)

    df['dis_sq'] = (df['dis_x'] + df['dis_y'] + df['dis_z']).astype(np.float32)
    df['dis_bond'] = np.sqrt(df['dis_sq'])
    df['dis_sq_inv'] = (1 / df['dis_sq']).astype(np.float32)
    return df


@timer('DistanceFeatures')
def _add_neighbor_xyz(df, structures_df):
    df.index.name = 'id'
    df = df.reset_index()
    df = pd.merge(
        df,
        structures_df,
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'],
        how='left')

    df.rename({'x': 'x_0', 'y': 'y_0', 'z': 'z_0', 'atom': 'atom_0'}, axis=1, inplace=True)

    df = pd.merge(
        df,
        structures_df,
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'],
        how='left')

    df.drop(['atom_index_x', 'atom_index_y'], axis=1, inplace=True)

    df.rename({'x': 'x_1', 'y': 'y_1', 'z': 'z_1', 'atom': 'atom_1'}, axis=1, inplace=True)
    df[['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']] = df[['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']].astype(np.float32)
    return df.set_index('id')


def add_distance_features(df, structures_df):
    df = _add_neighbor_xyz(df, structures_df)
    df = _add_bond_distance_feature(df)
    df = _add_distance_from_center(df, structures_df)
    _set_distance_from_molecule_centers_feature(df)
    return df
