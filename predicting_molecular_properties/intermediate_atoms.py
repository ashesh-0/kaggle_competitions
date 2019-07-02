"""
Here we find out for every pair in train and test data, the intermediate atoms
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np
from common_utils import find_distance_from_plane, find_distance_btw_point

EPSILON = 1e-5


def _add_plane_vector(df):
    df['m_x'] = (df['x_1'] - df['x_0']).values
    df['m_y'] = (df['y_1'] - df['y_0']).values
    df['m_z'] = (df['z_1'] - df['z_0']).values
    df['m_2norm'] = np.sqrt(df['m_x'] * df['m_x'] + df['m_y'] * df['m_y'] + df['m_z'] * df['m_z'])
    df['c'] = 0
    df['c'] = -1 * df['m_2norm'] * find_distance_from_plane(df, 'x_0', 'y_0', 'z_0').values
    df['x_1_distance'] = find_distance_from_plane(df, 'x_1', 'y_1', 'z_1').values


def _add_dist_frm_a_point_one_threshold(df, atom_enc, thresh, empty_index_df):

    df = df[df['atom_dis'] <= thresh]

    bond_dist_df = df.groupby('id')['atom_dis'].agg(['min', 'max', 'mean', 'std']).astype(np.float32)
    bond_dist_df.columns = [c + '_bond_dis_{}'.format(thresh) for c in bond_dist_df.columns]

    frac_bond_dist_df = bond_dist_df.divide(df.groupby('id')['dis_bond'].first(), axis=0).astype(np.float32)
    frac_bond_dist_df.columns = ['frac_{}_{}'.format(c, thresh) for c in frac_bond_dist_df]

    count_df = df.groupby(['id', 'atom_id'])['atom_index'].count().unstack().fillna(0)
    count_df.columns = atom_enc.inverse_transform(count_df.columns)
    count_df['total'] = count_df.sum(axis=1)
    count_df = count_df[['total', 'H']]
    count_df.columns = ['{}_{}'.format(c, thresh) for c in count_df.columns]
    output_df = pd.concat([count_df, bond_dist_df, frac_bond_dist_df], axis=1)

    output_df = output_df.join(empty_index_df, how='right').fillna(0)
    output_df[count_df.columns] = output_df[count_df.columns].astype(np.uint8)

    return output_df


def _add_stats_frm_atom_0(df, atom_enc, empty_index_df):
    df['atom_dis'] = find_distance_btw_point(df, 'x_0', 'y_0', 'z_0', 'x', 'y', 'z')
    outputs = []
    for thresh in [2, 3]:
        outputs.append(_add_dist_frm_a_point_one_threshold(df, atom_enc, thresh, empty_index_df))
    output_df = pd.concat(outputs, axis=1).fillna(0)
    output_df.columns = ['atom_0_{}'.format(c) for c in output_df.columns]
    return output_df


def _add_stats_frm_atom_1(df, atom_enc, empty_index_df):
    df['atom_dis'] = find_distance_btw_point(df, 'x_1', 'y_1', 'z_1', 'x', 'y', 'z')
    outputs = []
    for thresh in [2, 3]:
        outputs.append(_add_dist_frm_a_point_one_threshold(df, atom_enc, thresh, empty_index_df))

    output_df = pd.concat(outputs, axis=1).fillna(0)
    output_df.columns = ['atom_1_{}'.format(c) for c in output_df.columns]
    return output_df


def _add_stats_frm_bond_center(df, atom_enc, empty_index_df):
    df['bond_mid_x'] = (df['x_0'] + df['x_1']) / 2
    df['bond_mid_y'] = (df['y_0'] + df['y_1']) / 2
    df['bond_mid_z'] = (df['z_0'] + df['z_1']) / 2

    df['atom_dis'] = find_distance_btw_point(df, 'bond_mid_x', 'bond_mid_y', 'bond_mid_z', 'x', 'y', 'z')
    outputs = []
    for thresh in [2, 3]:
        outputs.append(_add_dist_frm_a_point_one_threshold(df, atom_enc, thresh, empty_index_df))

    output_df = pd.concat(outputs, axis=1).fillna(0)
    output_df.columns = ['bond_mid_{}'.format(c) for c in output_df.columns]
    return output_df


def _add_intermediate_counts_molecule_range(df, atom_enc, empty_index_df):
    """
    Count of number of molecules coming in between the two edges.
    """
    df['atom_dis'] = find_distance_from_plane(df, 'x', 'y', 'z')
    df.loc[df['atom_dis'].abs() < EPSILON, 'atom_dis'] = 0
    same_side_of_plane = df['atom_dis'] * df['x_1_distance'] >= 0
    close_enough = (df['atom_dis'].abs() <= df['x_1_distance'].abs())
    df['is_intermediate'] = same_side_of_plane & close_enough

    df = df[(df.is_intermediate == True)]
    output_df = df.groupby(['id', 'atom_id'])['atom_index'].count().unstack().fillna(0)
    output_df.columns = atom_enc.inverse_transform(output_df.columns)
    output_df['total'] = output_df.sum(axis=1)
    output_df.columns = [c + '_plane_cnt' for c in output_df.columns]
    output_df = output_df.join(empty_index_df, how='right').fillna(0).astype(np.uint8)
    return output_df


def _add_intermediate_stats_molecule_range(df, structures_df, start_mn, end_mn, x1_distance_fraction, atom_enc):
    structures_df = structures_df[(structures_df.molecule_id < end_mn) & (structures_df.molecule_id >= start_mn)]
    df = df[(df.molecule_id < end_mn) & (df.molecule_id >= start_mn)]

    # We want to compute features for all indices present in this empty df.
    empty_index_df = df.set_index('id')[[]]

    df = pd.merge(df, structures_df, how='left', on='molecule_id')
    if df.empty:
        return pd.DataFrame()

    # remove points corresponding to the two vertices.
    df = df[(df.atom_index != df.atom_index_0) & (df.atom_index != df.atom_index_1)]
    # counts of all molecules between the bonds
    cnt1_df = _add_intermediate_counts_molecule_range(df, atom_enc, empty_index_df)
    cnt2_df = _add_stats_frm_bond_center(df, atom_enc, empty_index_df)
    cnt3_df = _add_stats_frm_atom_0(df, atom_enc, empty_index_df)
    cnt4_df = _add_stats_frm_atom_1(df, atom_enc, empty_index_df)

    output_df = pd.concat([cnt1_df, cnt2_df, cnt3_df, cnt4_df], axis=1)
    return output_df


def add_intermediate_atom_stats(train_df, structures_df, step_size=1000, x1_distance_fraction=0.8):
    """
    Equation of plane normal to (x1-x0) is :m_x*x + m_y*y + m_z*z + c = 0
    We will select all points which lie between x_1 and x_0
    """

    assert len(set(['molecule_name', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(train_df.columns.tolist())) == 0
    df = train_df.reset_index()[[
        'id', 'molecule_name', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1', 'dis_bond', 'atom_index_0', 'atom_index_1'
    ]].copy()

    # replacing molecule_name by id.
    mn_enc = LabelEncoder()
    structures_df['molecule_id'] = mn_enc.fit_transform(structures_df['molecule_name'])
    df['molecule_id'] = mn_enc.fit_transform(df['molecule_name'])
    df.drop('molecule_name', axis=1, inplace=True)

    # replacing atom by atom_id
    atom_enc = LabelEncoder()
    structures_df['atom_id'] = atom_enc.fit_transform(structures_df['atom'])

    df[['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']] = df[['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']].astype(np.float32)

    # adding the equation of plane parameters.
    _add_plane_vector(df)

    output = []
    for start_mn in tqdm(range(0, structures_df.molecule_id.max(), step_size)):
        output.append(
            _add_intermediate_stats_molecule_range(
                df,
                structures_df,
                start_mn,
                start_mn + step_size,
                x1_distance_fraction,
                atom_enc,
            ))

    output_df = pd.concat(output)
    output_df.columns = ['DIS_' + c for c in output_df.columns]

    output_df = output_df.join(train_df[[]], how='right').fillna(0)
    return pd.concat([train_df, output_df], axis=1)
