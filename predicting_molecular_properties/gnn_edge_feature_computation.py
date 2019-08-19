import pandas as pd
import numpy as np

from common_utils_molecule_properties import dot, find_cross_product, find_projection_on_plane


###
def _add_edges(edges_df, ia_df, ai_0, ai_1, ai_2):
    """
    Find two edges ai_0-ai_1 and ai_1-ai_2
    """
    ia_df.reset_index(inplace=True)
    ia_df = pd.merge(
        ia_df,
        edges_df[['mol_id', 'atom_index_0', 'atom_index_1', 'x', 'y', 'z']],
        how='left',
        left_on=['mol_id', ai_0, ai_1],
        right_on=['mol_id', 'atom_index_0', 'atom_index_1'])

    ia_df.rename({'x': 'x_0', 'y': 'y_0', 'z': 'z_0'}, axis=1, inplace=True)
    ia_df.drop(['atom_index_0', 'atom_index_1'], axis=1, inplace=True)

    ia_df = pd.merge(
        ia_df,
        edges_df[['mol_id', 'atom_index_0', 'atom_index_1', 'x', 'y', 'z']],
        how='left',
        left_on=['mol_id', ai_1, ai_2],
        right_on=['mol_id', 'atom_index_0', 'atom_index_1'])

    ia_df.rename({'x': 'x_1', 'y': 'y_1', 'z': 'z_1'}, axis=1, inplace=True)
    ia_df.drop(['atom_index_0', 'atom_index_1'], axis=1, inplace=True)
    ia_df.set_index('id', inplace=True)
    return ia_df


def get_intermediate_angle_features(edges_df, X_df, structures_df, ia_df):
    cnt_df = (ia_df != -1).sum(axis=1)

    assert 'mol_id' in edges_df
    assert 'mol_id' in X_df
    assert 'mol_id' in structures_df
    assert 'mol_id' in ia_df

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

    feat_df = ia_df_3[['sin_out_of_plane', 'dihedral_angle', 'dihedral_angle_2@', 'angle_1']]
    feat_df = feat_df.join(X_df[[]], how='right')
    feat_df.loc[cnt_df == 3, 'dihedral_angle'] = feat_df.loc[cnt_df == 3, 'angle_1']

    return feat_df.astype(np.float16)
