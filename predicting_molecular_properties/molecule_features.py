"""
Features which are same for all atoms of a molecule.
"""
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
# from decorators import timer


@timer('MoleculeFeatures')
def unsaturation_count_features(structures_df):
    n_counts = structures_df.groupby(['molecule_name', 'atom']).size().unstack().fillna(0)
    nC = n_counts['C']
    nO = n_counts['O']
    nN = n_counts['N']
    nH = n_counts['H']
    nF = n_counts['F'] if 'F' in n_counts.columns else 0
    unsatu_cnt = ((2 * nC + 2 - 0 * nO + 1 * nN - 1 * nF) - nH) / 2
    unsatu_cnt = unsatu_cnt.to_frame('unsaturation_count').astype(np.float16)
    unsatu_cnt['unsaturation_fraction'] = unsatu_cnt['unsaturation_count'] / n_counts.sum(axis=1)
    # Not valid for non carbon atoms
    unsatu_cnt.loc[nC == 0, :] = -10
    return unsatu_cnt


@timer('MoleculeFeatures')
def count_features(structures_df):
    """
    How many Carbon, Oxygen and other atoms are present in the molecule.
    """
    atom_counts = structures_df.groupby(['molecule_name', 'atom'])['x'].count().unstack().fillna(0)
    atom_counts['total'] = atom_counts.sum(axis=1)
    atom_counts['non_H'] = atom_counts['total'] - atom_counts['H']
    atom_counts.columns = [c + '_count' for c in atom_counts.columns]
    return atom_counts.astype(np.int16)


@timer('MoleculeFeatures')
def convexhull_features(structures_df):
    ch_features = structures_df.groupby('molecule_name')[['x', 'y', 'z']].apply(_convexhull_features)
    ch_features = ch_features.to_frame('agg')
    ch_features['volume'] = ch_features['agg'].apply(lambda x: x[0])
    ch_features['area'] = ch_features['agg'].apply(lambda x: x[1])
    ch_features.drop('agg', inplace=True, axis=1)
    return ch_features.astype(np.float32)


@timer('MoleculeFeatures')
def molecule_shape_features(structures_df):
    min_xyz = structures_df.groupby('molecule_name')[['x', 'y', 'z']].min()
    max_xyz = structures_df.groupby('molecule_name')[['x', 'y', 'z']].max()
    quant_10_xyz = structures_df.groupby('molecule_name')[['x', 'y', 'z']].quantile(0.1)
    quant_50_xyz = structures_df.groupby('molecule_name')[['x', 'y', 'z']].quantile(0.5)

    feat1 = (max_xyz - min_xyz).rename({'x': 'x_len_1', 'y': 'y_len_1', 'z': 'z_len_1'}, axis=1)
    feat2 = min_xyz.rename({'x': 'x_len_0', 'y': 'y_len_0', 'z': 'z_len_0'}, axis=1)
    feat3 = (quant_10_xyz - min_xyz).rename({'x': 'x_len_.1', 'y': 'y_len_.1', 'z': 'z_len_.1'}, axis=1)
    feat4 = (quant_50_xyz - min_xyz).rename({'x': 'x_len_.5', 'y': 'y_len_.5', 'z': 'z_len_.5'}, axis=1)
    feat5 = feat1.divide(
        feat1.max(axis=1), axis=0).rename(
            {
                'x_len_1': 'x_ratio',
                'y_len_1': 'y_ratio',
                'z_len_1': 'z_ratio',
            }, axis=1)
    output_df = pd.concat([feat1, feat2, feat3, feat4, feat5], axis=1).astype(np.float32)
    output_df.columns = ['MolShape_' + c for c in output_df.columns]

    return output_df


def get_molecule_features(structures_df):
    count_f = count_features(structures_df)
    conhull_f = convexhull_features(structures_df)
    mol_shape_f = molecule_shape_features(structures_df)
    unsatur_f = unsaturation_count_features(structures_df)

    molecule_f = pd.concat([count_f, conhull_f, mol_shape_f, unsatur_f], axis=1)
    molecule_f['volume_normalized'] = (molecule_f['volume'] / molecule_f['total_count']).astype(np.float32)
    molecule_f['area_normalized'] = (molecule_f['area'] / molecule_f['total_count']).astype(np.float32)
    return molecule_f


def add_molecule_features(X_df, structures_df):
    mf = get_molecule_features(structures_df)
    X_df = X_df.join(mf, on='molecule_name', how='left')
    return X_df


# Private fns
def _convexhull_features(data):
    try:
        ch = ConvexHull(data)
    except:
        return [0, 0]

    return [ch.volume, ch.area]
