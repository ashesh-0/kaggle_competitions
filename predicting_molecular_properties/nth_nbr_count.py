"""
How many k neighbors does an atom has
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook


def compute_kneighbors_one_molecule(atom_types, edges_data, k_max, atom_types_dict):
    """
    atom_types:
    """
    n_atoms = atom_types.shape[0]
    adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
    adjacency_matrix[edges_data[:, 0], edges_data[:, 1]] = True
    adjacency_matrix[edges_data[:, 1], edges_data[:, 0]] = True

    atom_type_data = np.tile(atom_types.reshape(n_atoms, ), (n_atoms, 1))

    atom_type_mask = np.ones((n_atoms, n_atoms))
    # Ignore C,H
    atom_type_mask[atom_type_data == atom_types_dict['C']] = 0
    atom_type_mask[atom_type_data == atom_types_dict['H']] = 0
    k_nbr_matrix_list = [adjacency_matrix]

    for k in range(2, k_max + 1):
        k_nbr_matrix = k_nbr_matrix_list[k - 2] @ adjacency_matrix
        np.fill_diagonal(k_nbr_matrix, 0)
        k_nbr_matrix = k_nbr_matrix.astype(bool)

        # For k_th nbr computation, we can also find k-2th nbr as the edges are non directional.
        k_nbr_matrix[k_nbr_matrix_list[max(0, k - 3)]] = False
        k_nbr_matrix_list.append(k_nbr_matrix)

    features = np.zeros((n_atoms, k_max - 1))
    for k in range(1, k_max):
        features[:, k - 1] = (atom_type_mask * k_nbr_matrix_list[k]).sum(axis=1)

    return features


def compute_kneighbors_count(structures_df, edges_df, k_max=5):
    assert 'mol_id' in structures_df
    assert 'mol_id' in edges_df

    structures_df.sort_values(['mol_id', 'atom_index'], inplace=True)
    edges_df.sort_values(['mol_id'], inplace=True)

    enc = LabelEncoder()
    structures_df['atom_type'] = enc.fit_transform(structures_df['atom'])
    atom_types_dict = dict(zip(enc.classes_, enc.transform(enc.classes_)))

    start_structures_df_indices = structures_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    start_edges_df_indices = edges_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values

    edges_data = edges_df[['atom_index_0', 'atom_index_1']].values
    structures_data = structures_df[['atom_type']].values

    features = np.zeros((structures_data.shape[0], k_max - 1))
    for i, s_start_index in enumerate(tqdm_notebook(start_structures_df_indices)):
        s_end_index = start_structures_df_indices[
            i + 1] if i + 1 < len(start_structures_df_indices) else structures_data.shape[0]

        e_start_index = start_edges_df_indices[i]
        e_end_index = start_edges_df_indices[i + 1] if i + 1 < len(start_edges_df_indices) else edges_data.shape[0]
        features[s_start_index:s_end_index, :] = compute_kneighbors_one_molecule(
            structures_data[s_start_index:s_end_index],
            edges_data[e_start_index:e_end_index],
            k_max,
            atom_types_dict,
        )

    df = pd.DataFrame(features, index=structures_df.index)
    df['atom_index'] = structures_df['atom_index']
    df['mol_id'] = structures_df['mol_id']
    return df
