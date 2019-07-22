import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

# from bond_features import get_electonegativity
# from decorators import timer
# from common_utils import dot, get_structure_data
# from intermediate_atom_features import add_intermediate_atom_features


def get_symmetric_edges(edge_df):
    """
    Ensures that all edges in all molecules occur exactly twice in edge_df. This ensures that when we join with
    on with one of atom_index_0/atom_index_1, all edges are covered.
    """
    e_df = edge_df.copy()
    atom_1 = e_df.atom_index_1.copy()
    e_df['atom_index_1'] = e_df['atom_index_0']
    e_df['atom_index_0'] = atom_1
    e_df[['x', 'y', 'z']] = -1 * e_df[['x', 'y', 'z']]
    edge_df = pd.concat([edge_df, e_df], ignore_index=True)
    return edge_df


def add_electronetivity(df):
    """
    Adds electronegativity data for atom_0 and atom_1. It also adds difference of electronegativity.
    """
    assert 'atom_0' in df.columns
    assert 'atom_1' in df.columns
    electro_df = get_electonegativity()
    electro_df.index.name = 'atom_0'
    df = df.join(electro_df, how='left', on='atom_0')
    df.rename({'Electronegativity': 'Electronegativity_0'}, axis=1, inplace=True)

    electro_df.index.name = 'atom_1'
    df = df.join(electro_df, how='left', on='atom_1')
    df.rename({'Electronegativity': 'Electronegativity_1'}, axis=1, inplace=True)
    df['Electronegativity_diff'] = df['Electronegativity_1'] - df['Electronegativity_0']
    return df


def add_features_to_edges(edge_df, structures_df):
    """
    Electronegativity and atom are added for both atoms of the edge(atom_index_0,atom_index_1)
    """
    edge_df = pd.merge(
        edge_df,
        structures_df[['molecule_name', 'atom', 'atom_index']],
        how='left',
        left_on=['molecule_name', 'atom_index_0'],
        right_on=['molecule_name', 'atom_index'])

    edge_df.rename({'atom': 'atom_0'}, axis=1, inplace=True)
    edge_df.drop(['atom_index'], axis=1, inplace=True)

    edge_df = pd.merge(
        edge_df,
        structures_df[['molecule_name', 'atom', 'atom_index']],
        how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'])

    edge_df.rename({'atom': 'atom_1'}, axis=1, inplace=True)
    edge_df.drop(['atom_index'], axis=1, inplace=True)

    return add_electronetivity(edge_df)


def add_edge_features(edge_df, X_df, structures_df, ia_df, neighbors_df):
    """
    1. electonegativity feature for the bond.
    2. number of neighbors for both atoms of the bond.
    3. electronegativity vector for both atoms of the bond using their neighbors.
    """
    X_df = add_electronetivity_features(X_df)

    edge_df = get_symmetric_edges(edge_df)
    X_df = add_neighbor_count_features(edge_df, X_df, structures_df)
    add_bond_atom_aggregation_features(edge_df, X_df, structures_df, ia_df, neighbors_df)
    return X_df


def _induced_electronegativity_features(merged_df, df, atom_index):
    elec_df = merged_df.groupby('id')[['enegv_x', 'enegv_y', 'enegv_z']].sum()
    # bv: bond vector (for each row in df)
    elec_along_bond_df = dot(elec_df, df, ['enegv_x', 'enegv_y', 'enegv_z'], ['bondv_x', 'bondv_y', 'bondv_z']).dropna()
    elec_perp_bond_df = elec_df.pow(2).sum(axis=1).pow(0.5) - elec_along_bond_df.abs()

    elec_perp_bond_df = elec_perp_bond_df.to_frame(atom_index + '_induced_elecneg_perp').fillna(0)
    elec_along_bond_df = elec_along_bond_df.to_frame(atom_index + '_induced_elecneg_along').fillna(0)
    return pd.concat([elec_along_bond_df, elec_perp_bond_df], axis=1)


def _distance_features(merged_df, atom_index):
    distance_feature = merged_df.groupby('id').agg({'distance': ['min', 'max', 'mean', 'std']})['distance']
    distance_feature.columns = [atom_index + '_nbr_distance_' + c for c in distance_feature.columns]
    return distance_feature


def _angle_features(merged_df, atom_index):
    angle_with_neighbor_bonds = dot(merged_df, merged_df, ['bondv_x', 'bondv_y', 'bondv_z'],
                                    ['x', 'y', 'z']).to_frame('angle')
    angle_with_neighbor_bonds['id'] = merged_df['id']

    distance_feature = angle_with_neighbor_bonds.groupby('id').agg({'angle': ['min', 'max', 'mean', 'std']})['angle']
    distance_feature.columns = [atom_index + '_nbr_bond_angle_' + c for c in distance_feature.columns]
    return distance_feature


def _get_bond_atom_aggregation_features_one_atom(df, edge_df, atom_index):
    df_with_id = df.reset_index()
    mdf = pd.merge(
        df_with_id[['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'bondv_x', 'bondv_y', 'bondv_z']],
        edge_df,
        how='left',
        left_on=['molecule_name', atom_index],
        right_on=['molecule_name', 'atom_index_one'])
    if atom_index == 'atom_index_0':
        other_atom_index = 'atom_index_1'
    if atom_index == 'atom_index_1':
        other_atom_index = 'atom_index_0'

    # We don't want to include other atom of the bond as neighbor. In neighbors of atom_index_0, we don't want
    # atom_index_1
    mdf = mdf[mdf['atom_index_zero'] != mdf[other_atom_index]]
    electro_neg_features_df = _induced_electronegativity_features(mdf, df, atom_index)
    distance_features_df = _distance_features(mdf, atom_index)
    angle_features_df = _angle_features(mdf, atom_index)
    return pd.concat([electro_neg_features_df, distance_features_df, angle_features_df], axis=1)


@timer('EdgeFeatures')
def add_electronetivity_features(X_df):
    return add_electronetivity(X_df)


def add_kneighbor_aggregation_features(edge_df, X_df, structures_df, neighbors_df):
    feat_df = _get_kneighbor_aggregation_features(edge_df, X_df, structures_df, neighbors_df)
    X_df = X_df.reset_index()[['id', 'atom_index_0', 'atom_index_1', 'molecule_name']]

    cols = []
    for col in feat_df.columns:
        if col in ['molecule_name', 'atom_index']:
            cols.append(col)
        else:
            cols.append(f'ai0_{col}')

    feat_df.columns = cols

    X_df = pd.merge(
        X_df, feat_df, how='left', left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
    X_df.drop('atom_index', axis=1, inplace=True)

    cols = []
    for col in feat_df.columns:
        if col in ['molecule_name', 'atom_index']:
            cols.append(col)
        else:
            cols.append(f'ai1_{col[4:]}')

    feat_df.columns = cols
    X_df = pd.merge(
        X_df, feat_df, how='left', left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])

    X_df.drop('atom_index', axis=1, inplace=True)
    X_df.set_index('id', inplace=True)
    return X_df


def _get_kneighbor_aggregation_features(edge_df, X_df, structures_df, neighbors_df):
    neighbors_df = neighbors_df.join(X_df[['molecule_name']], how='left', on='id')
    neighbors_df.drop('id', inplace=True, axis=1)
    neighbors_df.index.name = 'id'
    neighbors_df.rename({'atom_index': 'atom_index_0', 'nbr_atom_index': 'atom_index_1'}, axis=1, inplace=True)
    temp_X_df = get_structure_data(neighbors_df, structures_df)
    feat_df = _add_bond_atom_aggregation_features(edge_df, temp_X_df, structures_df)
    feat_df['molecule_name'] = temp_X_df['molecule_name']
    feat_df['nbr_distance'] = temp_X_df['nbr_distance'].astype(np.uint8)
    feat_df['atom_index_0'] = temp_X_df['atom_index_0'].astype(np.uint8)
    feat_df = feat_df.groupby(['molecule_name', 'atom_index_0', 'nbr_distance']).mean().astype(np.float32)
    feat_df = feat_df.unstack().fillna(-10)
    feat_df.columns = [f'{nbr}NBR_{col}' for (col, nbr) in feat_df.columns]
    feat_df = feat_df.astype(np.float16)
    feat_df.reset_index(inplace=True)
    feat_df.rename({'atom_index_0': 'atom_index'}, inplace=True, axis=1)
    return feat_df


def add_kneighbor_along_path_aggregation_features(edge_df, X_df, structures_df, ia_df, k=1):
    """
    Gets electronegativity features for X,Y where X,Y are neighbors in path from atom1 to atom2. atom1--X-...-Y-atom2
    This is for k =1. for k=2,next to next neighbor is considered.
    """
    cnt_df = (~ia_df[ia_df != -1].isna()).sum(axis=1)
    filtr = cnt_df >= 3 + k - 1
    cnt_df = cnt_df[filtr]
    ia_df = ia_df[filtr]
    output = []
    for i in [0, 1]:
        temp_output = []
        for cnt in cnt_df.unique():
            temp_ia_df = ia_df[cnt_df == cnt].copy()
            # We compute features on atom1--X and Y--atom2 bonds in two turns of the for loop.
            temp_ia_df['atom_index_0'] = temp_ia_df[0] if i == 0 else temp_ia_df[cnt - 1]
            temp_ia_df['atom_index_1'] = temp_ia_df[k] if i == 0 else temp_ia_df[cnt - 1 - k]
            temp_ia_df['molecule_name'] = X_df.loc[temp_ia_df.index, 'molecule_name']

            temp_X_df = temp_ia_df[['atom_index_0', 'atom_index_1', 'molecule_name']]
            temp_X_df = get_structure_data(temp_X_df, structures_df)
            feat_df = _add_bond_atom_aggregation_features(edge_df, temp_X_df, structures_df)
            temp_output.append(feat_df)

        output_df = pd.concat(temp_output, axis=0)
        output_df = output_df.join(X_df[[]], how='right')
        feat_df = _fill_nan_aggregation_features(output_df)
        feat_df['EF_induced_elecneg_along_diff'] = feat_df['EF_induced_elecneg_along_diff'].fillna(0)
        feat_df.columns = [f'{k}nbr_ai{i}_{c}' for c in feat_df.columns]
        output.append(feat_df)

    return pd.concat(output, axis=1)


@timer('EdgeFeatures')
def add_bond_atom_aggregation_features(
        edge_df,
        X_df,
        structures_df,
        ia_df,
        neighbors_df,
        step=100,
):
    """
    For each row in X_df, we compute some features by aggregating on all edges for atom_index_0 and atom_index_1
    separately.
    """
    label = LabelEncoder()
    structures_df['m_id'] = label.fit_transform(structures_df['molecule_name'])
    X_df['m_id'] = label.transform(X_df['molecule_name'])
    edge_df['m_id'] = label.transform(edge_df['molecule_name'])
    edge_df = add_features_to_edges(edge_df, structures_df)
    output = []
    for start_m_id in tqdm_notebook(range(0, X_df['m_id'].max(), step)):
        st_t_df = structures_df[(structures_df['m_id'] >= start_m_id) & (structures_df['m_id'] < start_m_id + step)]
        X_t_df = X_df[(X_df['m_id'] >= start_m_id) & (X_df['m_id'] < start_m_id + step)]
        ia_t_df = ia_df.loc[X_t_df.index]
        edge_t_df = edge_df[(edge_df['m_id'] >= start_m_id) & (edge_df['m_id'] < start_m_id + step)]
        neighbors_t_df = neighbors_df[neighbors_df['id'].isin(X_t_df.index)]

        feat_ia_df = add_intermediate_atom_features(edge_t_df, X_t_df, st_t_df, ia_t_df)
        feat_df = _add_bond_atom_aggregation_features(edge_t_df, X_t_df, st_t_df)
        feat_ia2_df = add_kneighbor_along_path_aggregation_features(edge_t_df, X_t_df, st_t_df, ia_t_df, k=1)
        feat_ia3_df = add_kneighbor_along_path_aggregation_features(edge_t_df, X_t_df, st_t_df, ia_t_df, k=2)
        feat_nbr_df = add_kneighbor_aggregation_features(edge_t_df, X_t_df, st_t_df, neighbors_t_df)
        output.append(pd.concat([feat_df, feat_ia_df, feat_ia2_df, feat_ia3_df, feat_nbr_df], axis=1))

    for feat_df in output:
        for col in feat_df.columns:
            if col not in X_df.columns:
                X_df[col] = -10
            X_df.loc[feat_df.index, col] = feat_df[col]
            if feat_df.dtypes.loc[col] in [np.float64, np.float32, np.float16] and X_df.dtypes.loc[col] != np.float16:
                X_df[col] = X_df[col].astype(np.float16)

    X_df.drop('m_id', axis=1, inplace=True)
    edge_df.drop('m_id', axis=1, inplace=True)
    structures_df.drop('m_id', axis=1, inplace=True)


def _fill_nan_aggregation_features(feat_df):
    nan_with_0 = [
        'EF_atom_index_0_induced_elecneg_along',
        'EF_atom_index_0_induced_elecneg_perp',
        'EF_atom_index_1_induced_elecneg_along',
        'EF_atom_index_1_induced_elecneg_perp',
        'EF_atom_index_1_nbr_distance_std',
        'EF_atom_index_1_nbr_bond_angle_std',
    ]
    feat_df[nan_with_0] = feat_df[nan_with_0].fillna(0)

    nan_with_minus_10 = [
        'EF_atom_index_0_nbr_distance_mean',
        'EF_atom_index_0_nbr_bond_angle_mean',
        'EF_atom_index_1_nbr_distance_min',
        'EF_atom_index_1_nbr_distance_max',
        'EF_atom_index_1_nbr_distance_mean',
        'EF_atom_index_1_nbr_bond_angle_min',
        'EF_atom_index_1_nbr_bond_angle_max',
        'EF_atom_index_1_nbr_bond_angle_mean',
    ]
    feat_df[nan_with_minus_10] = feat_df[nan_with_minus_10].fillna(-10)
    return feat_df


def _add_bond_atom_aggregation_features(edge_df, X_df, structures_df):
    edge_df[['enegv_x', 'enegv_y', 'enegv_z']] = edge_df[['x', 'y', 'z']].multiply(
        edge_df['Electronegativity_diff'], axis=0)

    edge_df.rename(
        {
            'atom_index_0': 'atom_index_zero',
            'atom_index_1': 'atom_index_one',
        }, axis=1, inplace=True)
    df = X_df[['molecule_name', 'atom_index_0', 'atom_index_1']].copy()

    # bond vector
    df['bondv_x'] = X_df['x_1'] - X_df['x_0']
    df['bondv_y'] = X_df['y_1'] - X_df['y_0']
    df['bondv_z'] = X_df['z_1'] - X_df['z_0']

    # unit bond vector
    vlen = df[['bondv_x', 'bondv_y', 'bondv_z']].pow(2).sum(axis=1).pow(0.5)
    df[['bondv_x', 'bondv_y', 'bondv_z']] = df[['bondv_x', 'bondv_y', 'bondv_z']].divide(vlen, axis=0)

    feat_0 = _get_bond_atom_aggregation_features_one_atom(df, edge_df, 'atom_index_0')
    feat_1 = _get_bond_atom_aggregation_features_one_atom(df, edge_df, 'atom_index_1')

    feat = pd.concat([feat_0, feat_1], axis=1)

    feat.columns = ['EF_' + c for c in feat.columns]
    feat = _fill_nan_aggregation_features(feat)

    feat['EF_induced_elecneg_along_diff'] = (
        feat['EF_atom_index_1_induced_elecneg_along'] - feat['EF_atom_index_0_induced_elecneg_along'])

    feat = feat.astype(np.float16)

    # atom_index_0 is always H. So some features are not useful for it as it has just one neighbor.
    useless_features = [
        'EF_atom_index_0_nbr_distance_min',
        'EF_atom_index_0_nbr_distance_max',
        'EF_atom_index_0_nbr_distance_std',
        'EF_atom_index_0_nbr_bond_angle_min',
        'EF_atom_index_0_nbr_bond_angle_max',
        'EF_atom_index_0_nbr_bond_angle_std',
    ]
    feat.drop(useless_features, axis=1, inplace=True)
    return feat


@timer('EdgeFeatures')
def add_neighbor_count_features(edge_df, X_df, structures_df):
    """
    edge_df must be symmetric.
    """
    cnt_df = edge_df.groupby(['molecule_name', 'atom_index_0']).size().to_frame('EF_neighbor_count')
    cnt_df.reset_index(inplace=True)
    cnt_df.rename({'atom_index_0': 'atom_index_zero'}, inplace=True, axis=1)

    X_df = X_df.reset_index()
    X_df = pd.merge(
        X_df,
        cnt_df,
        how='left',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index_zero'])

    X_df.rename({'EF_neighbor_count': 'EF_atom_index_1_neighbor_count'}, inplace=True, axis=1)
    X_df.drop(['atom_index_zero'], inplace=True, axis=1)
    X_df.set_index('id', inplace=True)
    incorrect_absence = 100 * X_df['EF_atom_index_1_neighbor_count'].isna().sum() / X_df.shape[0]
    print('[EdgeFeatures] Setting following percentage of edges to 0:', incorrect_absence)

    X_df['EF_atom_index_1_neighbor_count'] = X_df['EF_atom_index_1_neighbor_count'].fillna(0).astype(np.uint8)
    return X_df
