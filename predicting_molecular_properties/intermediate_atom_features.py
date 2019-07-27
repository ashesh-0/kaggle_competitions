import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from bond_features import get_lone_pair

from common_utils_molecule_properties import (get_structure_data, dot, find_distance_btw_point, find_cross_product,
                                              find_projection_on_plane)


def add_intermediate_atom_features(edges_df, X_df, structures_df, ia_df):
    f1 = count_feature(ia_df)
    f2 = add_CC_hybridization_feature(ia_df, edges_df, X_df, structures_df)
    f3 = get_intermediate_angle_features(edges_df, X_df, structures_df, ia_df)

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

    return pd.concat([f1, f2[['CC_hybridization'] + oth_cols], f3], axis=1)


def _add_edges(edges_df, ia_df, ai_0, ai_1, ai_2):
    """
    Find two edges ai_0-ai_1 and ai_1-ai_2
    """
    ia_df.reset_index(inplace=True)
    ia_df = pd.merge(
        ia_df,
        edges_df[['m_id', 'atom_index_0', 'atom_index_1', 'x', 'y', 'z']],
        how='left',
        left_on=['m_id', ai_0, ai_1],
        right_on=['m_id', 'atom_index_0', 'atom_index_1'])

    ia_df.rename({'x': 'x_0', 'y': 'y_0', 'z': 'z_0'}, axis=1, inplace=True)
    ia_df.drop(['atom_index_0', 'atom_index_1'], axis=1, inplace=True)

    ia_df = pd.merge(
        ia_df,
        edges_df[['m_id', 'atom_index_0', 'atom_index_1', 'x', 'y', 'z']],
        how='left',
        left_on=['m_id', ai_1, ai_2],
        right_on=['m_id', 'atom_index_0', 'atom_index_1'])

    ia_df.rename({'x': 'x_1', 'y': 'y_1', 'z': 'z_1'}, axis=1, inplace=True)
    ia_df.drop(['atom_index_0', 'atom_index_1'], axis=1, inplace=True)
    ia_df.set_index('id', inplace=True)
    return ia_df


def get_intermediate_angle_features(edges_df, X_df, structures_df, ia_df):
    cnt_df = (ia_df != -1).sum(axis=1)
    enc = LabelEncoder()
    structures_df['m_id'] = enc.fit_transform(structures_df['molecule_name'])
    X_df['m_id'] = enc.transform(X_df['molecule_name'])
    edges_df['m_id'] = enc.transform(edges_df['molecule_name'])
    ia_df['m_id'] = X_df['m_id']

    # 4 atom based examples. 2 angles.
    ia_df_3 = ia_df[cnt_df >= 3].copy()
    ia_df_3 = _add_edges(edges_df, ia_df_3, 0, 1, 2)
    ia_df_3['angle_1'] = dot(ia_df_3, ia_df_3, ['x_0', 'y_0', 'z_0'], ['x_1', 'y_1', 'z_1'])

    # normal to the plane containing atoms 0,1 and 2
    plane_con_01_edges = find_cross_product(ia_df_3, 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1')
    planeA_perp_1_edge = ia_df_3[['x_1', 'y_1', 'z_1']].copy()
    edge0_on_planeA = find_projection_on_plane(ia_df_3, planeA_perp_1_edge, 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1')

    ia_df_3.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1, inplace=True)
    # 2nd angle.
    ia_df_3 = _add_edges(edges_df, ia_df_3, 1, 2, 3)
    ia_df_3['angle_2'] = dot(ia_df_3, ia_df_3, ['x_0', 'y_0', 'z_0'], ['x_1', 'y_1', 'z_1'])
    # angle made by 3rd edge from the plane of first two edges. It either side should not matter. so an abs().
    ia_df_3['sin_out_of_plane'] = dot(plane_con_01_edges, ia_df_3, ['x', 'y', 'z'], ['x_1', 'y_1', 'z_1']).abs()
    ia_df_3['sin_out_of_plane_2@'] = 2 * ia_df_3['sin_out_of_plane'].pow(2) - 1

    edge2_on_planeA = find_projection_on_plane(ia_df_3, planeA_perp_1_edge, 'x_1', 'y_1', 'z_1', 'x_1', 'y_1', 'z_1')
    ia_df_3['dihedral_angle'] = dot(edge0_on_planeA, edge2_on_planeA, ['x', 'y', 'z'], ['x', 'y', 'z'])
    ia_df_3['dihedral_angle_2@'] = 2 * ia_df_3['dihedral_angle'].pow(2) - 1

    feat_df = ia_df_3[[
        'angle_1', 'angle_2', 'sin_out_of_plane', 'dihedral_angle', 'dihedral_angle_2@', 'sin_out_of_plane_2@'
    ]]
    feat_df = feat_df.join(X_df[[]], how='right').fillna(-10)
    feat_df.loc[cnt_df == 3, 'dihedral_angle'] = feat_df.loc[cnt_df == 3, 'angle_1']

    structures_df.drop('m_id', axis=1, inplace=True)
    X_df.drop('m_id', axis=1, inplace=True)
    edges_df.drop('m_id', axis=1, inplace=True)
    ia_df.drop('m_id', axis=1, inplace=True)
    return feat_df.astype(np.float16)


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
