"""
Neighbors of atom_0 and atom_1 are inferred using distance between atoms and them and using standard bond lengths.
Features are computed on Neighbors.
"""
import numpy as np
import pandas as pd
from bond_features import get_bond_data
from common_utils_molecule_properties import find_distance_btw_point
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook


def bond_neighbor_features(X_df, str_df):
    """
    1. How many neighbors are present for both indices.
    2. Average bond length
    3. How average bond length compares with the bond in consideration.
    """
    assert len(
        set(['molecule_id', 'atom_1', 'atom_0', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(X_df.columns)) == 0
    assert 'molecule_id' in str_df.columns


def _get_features(df, atom_enc, bond_len_factor, for_atom_0, empty_df_index):
    # for each row in df, it added a 0/1 value in bond_count column denoting whether a bond exists between (x,y,z) and
    # one of [atom_0,atom_1] depending upon for_atom_0
    df = _get_raw_features(df, atom_enc, bond_len_factor, for_atom_0)
    # Select the neighbors and compute stats on them.
    df = df[df['bond_count'] == 1]
    features_df = df.groupby('id').agg({
        'bond_count': 'sum',
        'bond_type': ['max', 'mean'],
        'Electronegativity_diff': ['max', 'min', 'mean'],
    })
    features_df = features_df.join(empty_df_index[[]], how='right').fillna(0)
    prefix = 'Atom0_Neigbor_' if for_atom_0 else 'Atom1_Neigbor_'
    features_df.columns = [prefix + '_'.join(col).strip() for col in features_df.columns.values]

    return features_df


def _get_raw_features(df, atom_enc, bond_len_factor, for_atom_0):
    assert 'standard_bond_length' not in df.columns

    bond_data_df = get_bond_data(return_limited=False)[[
        'atom_0',
        'atom_1',
        'bond_type',
        'standard_bond_length',
        'Electronegativity_diff',
        # 'atom_0_Electronegativity',
        # 'atom_1_Electronegativity',
    ]]
    bond_data_df['atom_0'] = atom_enc.transform(bond_data_df['atom_0']).astype(np.uint8)
    bond_data_df['atom_1'] = atom_enc.transform(bond_data_df['atom_1']).astype(np.uint8)

    bond_data_df.rename({'atom_0': 'atom_zero', 'atom_1': 'atom_one'}, axis=1, inplace=True)

    # It will be True for rows when bond exists between atom_0 and some other atom in the molecule
    df['bond_count'] = 0
    if for_atom_0:
        # distance of all atoms in the molecule from atom_0
        dis = find_distance_btw_point(df, 'x_0', 'y_0', 'z_0', 'x', 'y', 'z').astype(np.float32)
        df = pd.merge(df, bond_data_df, how='left', left_on=['atom_0', 'atom'], right_on=['atom_zero', 'atom_one'])
    else:
        # distance of all atoms in the molecule from atom_0
        dis = find_distance_btw_point(df, 'x_1', 'y_1', 'z_1', 'x', 'y', 'z').astype(np.float32)
        df = pd.merge(df, bond_data_df, how='left', left_on=['atom_1', 'atom'], right_on=['atom_zero', 'atom_one'])

    df.loc[dis <= bond_len_factor * df['standard_bond_length'], 'bond_count'] = 1
    return df


def _add_intermediate_stats_molecule_range(
        df,
        structures_df,
        start_mn,
        end_mn,
        atom_enc,
        bond_len_factor=1.2,
):
    """
    bond_len_factor: this times the standard bond length is the max distance between the atoms for us to consider them as bonded.
    """

    structures_df = structures_df[(structures_df.molecule_id < end_mn) & (structures_df.molecule_id >= start_mn)]
    df = df[['id', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'atom_0', 'atom_1', 'molecule_id']]
    df = df[(df.molecule_id < end_mn) & (df.molecule_id >= start_mn)]

    df = pd.merge(df, structures_df, how='left', on='molecule_id')

    # remove points corresponding to the two vertices.
    df = df[(df.atom_index != df.atom_index_0) & (df.atom_index != df.atom_index_1)]
    if df.empty:
        return pd.DataFrame()

    df['atom'] = atom_enc.transform(df['atom'])

    atom_0_features = _get_features(df, atom_enc, bond_len_factor, True)
    atom_1_features = _get_features(df, atom_enc, bond_len_factor, False)
    return pd.concat([atom_0_features, atom_1_features], axis=1)


def add_neighbor_features_based_on_distance(df, structures_df, atom_enc, bond_len_factor=1.05, step_size=1000):

    assert len(set(['molecule_name', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(df.columns.tolist())) == 0
    df = df.reset_index()[[
        'id', 'molecule_name', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1', 'dis_bond', 'atom_index_0', 'atom_index_1'
    ]].copy()

    # replacing molecule_name by id.
    mn_enc = LabelEncoder()
    structures_df['molecule_id'] = mn_enc.fit_transform(structures_df['molecule_name'])
    df['molecule_id'] = mn_enc.fit_transform(df['molecule_name'])
    df.drop('molecule_name', axis=1, inplace=True)

    # replacing atom by atom_id

    structures_df['atom_id'] = atom_enc.fit_transform(structures_df['atom'])

    df[['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']] = df[['x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']].astype(np.float32)

    output = []
    for start_mn in tqdm_notebook(range(0, structures_df.molecule_id.max(), step_size)):
        output.append(
            _add_intermediate_stats_molecule_range(
                df, structures_df, start_mn, start_mn + step_size, atom_enc, bond_len_factor=bond_len_factor))

    output_df = pd.concat(output)
    output_df.columns = ['NF_Dis_' + c for c in output_df.columns]

    output_df = output_df.join(df[[]], how='right').fillna(0)
    return pd.concat([df, output_df], axis=1)
