import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from bond_features import get_lone_pair

from common_utils_molecule_properties import get_structure_data, dot, find_distance_btw_point


def add_intermediate_atom_features(edges_df, X_df, structures_df, ia_df):
    f1 = count_feature(ia_df)
    f2 = add_CC_hybridization_feature(ia_df, edges_df, X_df, structures_df)
    hyb_cols = ['CC_sp2_hyb', 'CC_sp3_hyb', 'CC_sp_hyb', 'CC_sp3-sp_hyb', 'CC_sp3-sp2_hyb', 'CC_sp2-sp_hyb']
    oth_cols = list(set(f2.columns) - set(hyb_cols))
    hyb_df = f2[hyb_cols]
    hyb_df.columns.name = 'CC_hybridization'
    hyb_df = hyb_df[hyb_df == 1].stack().reset_index(level=1)[['CC_hybridization']]
    enc = LabelEncoder()
    enc.classes_ = hyb_cols

    hyb_df['CC_hybridization'] = enc.transform(hyb_df['CC_hybridization'])
    f2['CC_hybridization'] = hyb_df['CC_hybridization']
    f2['CC_hybridization'] = f2['CC_hybridization'].fillna(-1)

    return pd.concat([f1, f2[['CC_hybridization'] + oth_cols]], axis=1)


def count_feature(ia_df):
    df = (ia_df != -1).sum(axis=1) - 2
    return df.to_frame('IA_count')


def add_CC_hybridization_feature(ia_df, edges_df, X_df, structures_df):
    """
    Features related to following structure: atom1--C--C--atom2.
    1. CYS-TRANS determination
    2. hybridization of the carbon.
    """
    atom_df = structures_df.set_index(['molecule_name', 'atom_index'])[['atom']]
    # symmetric assumed.
    edge_count_df = edges_df.groupby(['molecule_name', 'atom_index_0']).size().to_frame('edge_count').reset_index()
    edge_count_df.rename({'atom_index_0': 'atom_index'}, inplace=True, axis=1)
    edge_count_df.set_index(['molecule_name', 'atom_index'], inplace=True)
    atom_feature = pd.concat([atom_df, edge_count_df], axis=1).reset_index()
    atom_feature['edge_count'] = atom_feature['edge_count'].fillna(0)
    assert not atom_feature.isna().any().any()

    df = count_feature(ia_df)
    df['molecule_name'] = X_df['molecule_name']
    df['atom_index_1'] = ia_df[1]
    df['atom_index_0'] = ia_df[2]
    df = df[df['IA_count'] == 2].copy()
    df = df.reset_index()

    df = pd.merge(
        df,
        atom_feature,
        how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'])
    df.rename({'edge_count': 'edge_count_nbr_1', 'atom': 'atom_nbr_1'}, inplace=True, axis=1)
    df.drop(['atom_index'], axis=1, inplace=True)

    df = pd.merge(
        df,
        atom_feature,
        how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'])
    df.rename({'edge_count': 'edge_count_nbr_0', 'atom': 'atom_nbr_0'}, inplace=True, axis=1)
    df.drop(['atom_index'], axis=1, inplace=True)

    df['lone_pair_nbr_1'] = df['atom_nbr_1'].map(get_lone_pair()).astype(np.float16)
    df['lone_pair_nbr_0'] = df['atom_nbr_0'].map(get_lone_pair()).astype(np.float16)

    df['CC_interm'] = (df['atom_nbr_1'] == 'C') & (df['atom_nbr_0'] == 'C')
    # sp2,sp3,sp
    df['CC_sp_hyb'] = ((df['edge_count_nbr_1'] == 2) & (df['edge_count_nbr_0'] == 2)).astype(np.int8)
    df['CC_sp2_hyb'] = ((df['edge_count_nbr_1'] == 3) & (df['edge_count_nbr_0'] == 3)).astype(np.int8)
    df['CC_sp3_hyb'] = ((df['edge_count_nbr_1'] == 4) & (df['edge_count_nbr_0'] == 4)).astype(np.int8)

    df['CC_sp3-sp2_hyb'] = ((df['edge_count_nbr_1'] == 4) & (df['edge_count_nbr_0'] == 3))
    df['CC_sp3-sp2_hyb'] = df['CC_sp3-sp2_hyb'] | ((df['edge_count_nbr_1'] == 3) & (df['edge_count_nbr_0'] == 4))
    df['CC_sp3-sp2_hyb'] = df['CC_sp3-sp2_hyb'].astype(np.int8)

    df['CC_sp3-sp_hyb'] = ((df['edge_count_nbr_1'] == 4) & (df['edge_count_nbr_0'] == 2))
    df['CC_sp3-sp_hyb'] = df['CC_sp3-sp_hyb'] | ((df['edge_count_nbr_1'] == 2) & (df['edge_count_nbr_0'] == 4))
    df['CC_sp3-sp_hyb'] = df['CC_sp3-sp_hyb'].astype(np.int8)

    df['CC_sp2-sp_hyb'] = ((df['edge_count_nbr_1'] == 3) & (df['edge_count_nbr_0'] == 2))
    df['CC_sp2-sp_hyb'] = df['CC_sp2-sp_hyb'] | ((df['edge_count_nbr_1'] == 2) & (df['edge_count_nbr_0'] == 3))
    df['CC_sp2-sp_hyb'] = df['CC_sp2-sp_hyb'].astype(np.int8)

    df.loc[df['CC_interm'] == False,
           ['CC_sp2_hyb', 'CC_sp3_hyb', 'CC_sp_hyb', 'CC_sp2-sp_hyb', 'CC_sp3-sp_hyb', 'CC_sp3-sp2_hyb']] = -1

    # cys-trans
    # atom1--C==C--atom2
    # Image of atom1 in plane perpendicular  to C==C should be close to atom2 for cys.
    # Image is x0 +2*alpha
    # c = ['molecule_name', 'atom_index_0', 'atom_index_1']
    df.set_index('id', inplace=True)
    df = get_structure_data(df, structures_df)

    df.reset_index(inplace=True)
    df = pd.merge(df, edges_df, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1'])

    df['x_mid'] = (df['x_0'] + df['x_1']) / 2
    df['y_mid'] = (df['y_0'] + df['y_1']) / 2
    df['z_mid'] = (df['z_0'] + df['z_1']) / 2

    df['mc'] = -1 * dot(df, df, ['x', 'y', 'z'], ['x_mid', 'y_mid', 'z_mid'])

    df.rename({'x': 'mx', 'y': 'my', 'z': 'mz'}, axis=1, inplace=True)

    df.set_index('id', inplace=True)

    # removing C==C xyz coordinates.
    df.drop(
        ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_mid', 'y_mid', 'z_mid', 'atom_index_1', 'atom_index_0'],
        axis=1,
        inplace=True)

    # adding atom0,atom1 xyz coordinates.
    for c in ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1']:
        df[c] = X_df.loc[df.index, c]

    # since mx,my,mz is normalized vector.
    alpha = -1 * (dot(df, df, ['x_0', 'y_0', 'z_0'], ['mx', 'my', 'mz']) + df['mc'])

    df['x_1_ap'] = df['x_0'] + 2 * alpha * df['mx']
    df['y_1_ap'] = df['y_0'] + 2 * alpha * df['my']
    df['z_1_ap'] = df['z_0'] + 2 * alpha * df['mz']
    df['CC_cys_dis'] = find_distance_btw_point(df, 'x_1_ap', 'y_1_ap', 'z_1_ap', 'x_1', 'y_1', 'z_1')
    df.loc[df['CC_interm'] == False, 'CC_cys_dis'] = -1
    df.loc[df['CC_sp2_hyb'] == False, 'CC_cys_dis'] = -1

    df = df[[
        'CC_interm',
        'CC_cys_dis',
        'CC_sp2_hyb',
        'CC_sp3_hyb',
        'CC_sp_hyb',
        'CC_sp3-sp_hyb',
        'CC_sp3-sp2_hyb',
        'CC_sp2-sp_hyb',
    ]].join(
        X_df[[]], how='right')

    df[[
        'CC_sp2_hyb',
        'CC_sp3_hyb',
        'CC_sp_hyb',
        'CC_sp3-sp_hyb',
        'CC_sp3-sp2_hyb',
        'CC_sp2-sp_hyb',
    ]] = df[[
        'CC_sp2_hyb',
        'CC_sp3_hyb',
        'CC_sp_hyb',
        'CC_sp3-sp_hyb',
        'CC_sp3-sp2_hyb',
        'CC_sp2-sp_hyb',
    ]].fillna(-1).astype(np.int16)
    df['CC_cys_dis'] = df['CC_cys_dis'].fillna(-1).astype(np.float16)

    df['CC_interm'] = df['CC_interm'].fillna(False)
    return df
