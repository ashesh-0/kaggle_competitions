"""
Given an atom, I'll create features to segment
"""
import numpy as np
import itertools
from tqdm import tqdm_notebook
import pandas as pd

from gnn_common_utils_molecule_properties import compute_bond_vector_one_molecule


def compute_distance_feature_one_molecule(xyz_data, k_nearest_distances):
    """
    First 3 columns are x,y,z
    Distance to nearest 3 atoms.
    """

    n_atoms = xyz_data.shape[0]
    # 3rd dimension is replication.
    distances = compute_bond_vector_one_molecule(xyz_data)['bond_length']
    distances = 1 / distances
    sorted_atoms = distances.argsort()
    output = np.zeros((n_atoms, k_nearest_distances))
    for i in range(n_atoms):
        output[i, :min(k_nearest_distances, n_atoms)] = distances[i][sorted_atoms[i]][-k_nearest_distances:]

    return output


def compute_segmented_feature_one_molecule(molecule_data, distance_regions):
    """
    First 3 columns are x,y,z
    """

    distance_regions = [0] + distance_regions

    n_atoms = molecule_data.shape[0]
    # 3rd dimension is replication.
    distances = compute_bond_vector_one_molecule(molecule_data[:, :3])['bond_length']
    features = np.tile(molecule_data[:, 3:], (n_atoms, 1, 1))
    # features = features / distances.reshape((n_atoms, n_atoms, 1))
    output_arr = []
    for i in range(1, len(distance_regions)):
        mask = (distances >= distance_regions[i - 1]) * (distances < distance_regions[i])
        feat = (features * mask.reshape(n_atoms, n_atoms, 1)).sum(axis=1)
        output_arr.append(feat)

    return np.hstack(output_arr)


def compute_spherically_segmented_features(structures_df, regions=[2, 5]):
    """
    Count features
    """
    lone_pair = {'N': 1, 'O': 2, 'F': 3, 'H': 0, 'C': 0}
    valence_electrons = {'N': 5, 'O': 6, 'F': 7, 'H': 1, 'C': 4}
    structures_df['LonePairAtom'] = structures_df['atom'].map(lone_pair)
    structures_df['ValenceElectronsAtom'] = structures_df['atom'].map(valence_electrons)
    structures_df['CountAtom'] = 1

    feature_columns = ['LonePairAtom', 'ValenceElectronsAtom', 'CountAtom']
    structures_df.sort_values(['molecule_name', 'atom_index'], inplace=True)

    mol_idx_df = structures_df.groupby('molecule_name').size().cumsum()
    mol_start_idx_df = mol_idx_df.shift(1).fillna(0).astype(int)
    molecule_start_indices = mol_start_idx_df.values

    molecule_data = structures_df[['x', 'y', 'z', 'LonePairAtom', 'ValenceElectronsAtom', 'CountAtom']].values
    features = np.zeros((molecule_data.shape[0], len(regions) * len(feature_columns)), dtype=np.float16)
    for i, m_start_index in tqdm_notebook(list(enumerate(molecule_start_indices))):
        m_end_index = molecule_start_indices[i + 1] if i + 1 < len(molecule_start_indices) else molecule_data.shape[0]

        mol_features = compute_segmented_feature_one_molecule(molecule_data[m_start_index:m_end_index], regions)
        features[m_start_index:m_end_index] = mol_features

    columns = [f'AtomF_{feat}_{dis}' for dis, feat in itertools.product(regions, feature_columns)]

    return pd.DataFrame(features, columns=columns, index=structures_df.index)


def compute_knearest_distance_features(structures_df, k=3):
    """
    Count features
    """
    structures_df.sort_values(['molecule_name', 'atom_index'], inplace=True)

    mol_idx_df = structures_df.groupby('molecule_name').size().cumsum()
    mol_start_idx_df = mol_idx_df.shift(1).fillna(0).astype(int)
    molecule_start_indices = mol_start_idx_df.values

    molecule_data = structures_df[['x', 'y', 'z']].values
    features = np.zeros((molecule_data.shape[0], k), dtype=np.float16)
    for i, m_start_index in tqdm_notebook(list(enumerate(molecule_start_indices))):
        m_end_index = molecule_start_indices[i + 1] if i + 1 < len(molecule_start_indices) else molecule_data.shape[0]

        mol_features = compute_distance_feature_one_molecule(molecule_data[m_start_index:m_end_index], k)
        features[m_start_index:m_end_index] = mol_features

    columns = [f'AtomF_{ith}NearestDistance' for ith in reversed(range(k))]

    return pd.DataFrame(features, columns=columns, index=structures_df.index)


def get_spherically_symmetric_features(structures_df):
    df1 = compute_knearest_distance_features(structures_df)
    df2 = compute_spherically_segmented_features(structures_df)
    df = pd.concat([df1, df2], axis=1)
    assert df.index.equals(structures_df.index)
    df['mol_id'] = structures_df['mol_id']
    df['atom_index'] = structures_df['atom_index']

    return df
