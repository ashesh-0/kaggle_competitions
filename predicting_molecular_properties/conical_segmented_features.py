"""
Figure: Yi--A--B--Xi

For each A--B coupling in train df, we find what angles all other atoms Yi make from A--B bond on A as center.
Similarly we find Xi with B as center. We discritize the angle so as to get conical regions. We aggregate within a
conical region
Aggregation is done of
    1. Induced charge.
    2. Electronegativity.
    2. Mass.
    3. Count.
    4. Lone pairs.
    5. Electron count in outermost orbit.

"""
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

from common_data_molecule_properties import get_electonegativity

# from atom_potentials import compute_induced_charge_on_atoms
# from decorators import timer


def compute_feature_one_molecule(train_data_ai_0: np.array,
                                 train_data_ai_1: np.array,
                                 molecule_data: np.array,
                                 angle_segmentation_factor,
                                 hard_cutoff_distance=6):
    """
    molecule_data: Columns are x,y,z, Induced charge, mass, #lone pairs, #electrons in outermost shell. It is ordered by
                atom_index
    train_data_ai_0: Columns are x,y,z, Induced charge, mass, #lone pairs, #electrons in outermost shell.
                    For atom_index_0
    train_data_ai_1: Columns are x,y,z, Induced charge, mass, #lone pairs, #electrons in outermost shell.
                    For atom_index_1

    """
    atom_count = molecule_data.shape[0]
    train_count = train_data_ai_0.shape[0]

    xyz = [0, 1, 2]

    atom_index_idx = 3

    bond_vector_0 = train_data_ai_0[:, xyz] - train_data_ai_1[:, xyz]
    bond_vector_0 = bond_vector_0 / np.linalg.norm(bond_vector_0, axis=1).reshape(train_count, 1)

    A = np.tile(bond_vector_0.T, (atom_count, 1, 1)).T

    bond_mid_xyz = (train_data_ai_0[:, xyz] + train_data_ai_1[:, xyz]) / 2
    C = np.tile(bond_mid_xyz.T, (atom_count, 1, 1)).T

    B = np.tile(molecule_data.T, (train_count, 1, 1))

    B[:, xyz, :] = B[:, xyz, :] - C[:, xyz, :]
    YiA_distance = np.linalg.norm(B[:, xyz, :], axis=1)
    # Ignore the two atoms of the bond.
    YiA_distance[np.arange(train_count), train_data_ai_0[:, atom_index_idx].astype(int)] = 1e10
    YiA_distance[np.arange(train_count), train_data_ai_1[:, atom_index_idx].astype(int)] = 1e10
    # Set this to ensure that longer bond distances are not used.
    YiA_distance[YiA_distance > hard_cutoff_distance] = 1e10
    # xyz will become unit vectors, features will get normalized by distance. large distance will have very low
    # contributions
    B[:, xyz, :] = B[:, xyz, :] / YiA_distance.reshape(B.shape[0], 1, B.shape[2])
    B[:, 3:, :] = B[:, 3:, :] / np.power(YiA_distance, 3).reshape(B.shape[0], 1, B.shape[2])

    angle = (A[:, xyz, :] * B[:, xyz, :]).sum(axis=1)

    # Discretize angle into integer segments.
    angle = (angle * angle_segmentation_factor).astype(int)

    features = []
    for region in range(-angle_segmentation_factor + 1, angle_segmentation_factor, 1):
        filtr = (angle == region).astype(int)
        filtr = filtr.reshape((B.shape[0], 1, B.shape[2]))
        feature_one_region = (B[:, 3:, :] * filtr).sum(axis=2)
        features.append(feature_one_region)

    return np.hstack(features)


# @timer('ConicalSegmentedFeatures')
def add_conical_segmented_feature(X_df, structures_df, edges_df, angle_segmentation_factor=4):
    electroneg_df = get_electonegativity()
    mass = {'N': 14, 'O': 16, 'F': 19, 'H': 1, 'C': 12}
    lone_pair = {'N': 1, 'O': 2, 'F': 3, 'H': 0, 'C': 0}
    valence_electrons = {'N': 5, 'O': 6, 'F': 7, 'H': 1, 'C': 4}

    structures_df = compute_induced_charge_on_atoms(structures_df, edges_df)
    structures_df['EN'] = structures_df['atom'].map(electroneg_df)
    structures_df['MassAtom'] = structures_df['atom'].map(mass)
    structures_df['LonePairAtom'] = structures_df['atom'].map(lone_pair)
    structures_df['ValenceElectronsAtom'] = structures_df['atom'].map(valence_electrons)
    structures_df['CountAtom'] = 1

    feature_columns = ['EN', 'MassAtom', 'LonePairAtom', 'ValenceElectronsAtom', 'CountAtom', 'Q', 'Q4']

    structures_df = structures_df[structures_df.molecule_name.isin(X_df['molecule_name'].unique())]
    structures_df.sort_values(['molecule_name', 'atom_index'], inplace=True)

    X_df.sort_values(['molecule_name'], inplace=True)

    train_idx_df = X_df.groupby('molecule_name').size().cumsum()
    train_start_idx_df = train_idx_df.shift(1).fillna(0).astype(int)
    train_start_indices = train_start_idx_df.values

    mol_idx_df = structures_df.groupby('molecule_name').size().cumsum()
    mol_start_idx_df = mol_idx_df.shift(1).fillna(0).astype(int)
    molecule_start_indices = mol_start_idx_df.values

    assert train_start_indices.shape[0] == molecule_start_indices.shape[0]

    molecule_data = structures_df[[
        'x', 'y', 'z', 'EN', 'MassAtom', 'LonePairAtom', 'ValenceElectronsAtom', 'CountAtom', 'Q', 'Q4'
    ]].values
    train_data_ai_0 = X_df[['x_0', 'y_0', 'z_0', 'atom_index_0']].values
    train_data_ai_1 = X_df[['x_1', 'y_1', 'z_1', 'atom_index_1']].values

    features = _get_features(train_data_ai_0, train_data_ai_1, molecule_start_indices, molecule_data,
                             train_start_indices, angle_segmentation_factor)

    features_df = _get_df(features, angle_segmentation_factor, feature_columns, X_df.index)
    for col in features_df:
        X_df[col] = features_df[col]


def _get_features(train_data_ai_0, train_data_ai_1, molecule_start_indices, molecule_data, train_start_indices,
                  angle_segmentation_factor):
    """
    Computes the features, one molecule at a time.
    """
    features = np.zeros((train_data_ai_0.shape[0], 7 * 7), dtype=np.float16)
    for i, m_start_index in tqdm_notebook(list(enumerate(molecule_start_indices))):
        m_end_index = molecule_start_indices[i + 1] if i + 1 < len(molecule_start_indices) else molecule_data.shape[0]
        t_start_index = train_start_indices[i]
        t_end_index = train_start_indices[i + 1] if i + 1 < len(train_start_indices) else train_data_ai_1.shape[0]

        mol_features = compute_feature_one_molecule(
            train_data_ai_0[t_start_index:t_end_index],
            train_data_ai_1[t_start_index:t_end_index],
            molecule_data[m_start_index:m_end_index],
            angle_segmentation_factor,
        )
        features[t_start_index:t_end_index] = mol_features
    return features


def _get_df(features: np.array, angle_segmentation_factor, feature_columns, X_df_index):
    """
    From features present as numpy array, return a dataframe with appropriate column names.
    """
    columns = []
    formt = 'CONIC_REGION_{}_{}'
    for conic_region in range(-angle_segmentation_factor + 1, angle_segmentation_factor, 1):
        for feature in feature_columns:
            columns.append(formt.format(conic_region, feature))

    features_df = pd.DataFrame(features, index=X_df_index, columns=columns, dtype=np.float32)
    return features_df
