"""
Fast computation of angles between multiple tuples of 3 atoms.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

from gnn_common_utils_molecule_properties import compute_neighbor_mask, compute_bond_vector_one_molecule


def compute_angle_one_molecule(xyz_data, atom_index_data, neighbor_mask):
    """
    A--O--B
    Each row in atom_index_data contains one A--O
    Here, We find angle AOB for each row, for each B belonging to nearest 3 neighbors.
    Args:
        neighbor_mask: 29*29 boolean array specifying which are the neighbors
    """
    data = compute_bond_vector_one_molecule(xyz_data)
    bond_vector = data['bond_vector']
    n_atoms = xyz_data.shape[0]
    neighbor_mask = neighbor_mask[:n_atoms, :n_atoms]

    # For B, we don't want to select either A or O. Hence the -2
    # neighbor_count = min(neighbor_count, n_atoms - 2)

    AO_vector = bond_vector[atom_index_data[:, 0], :, atom_index_data[:, 1]].T
    # (nbrs,xyz,n_entries)
    AO_vector = np.tile(AO_vector, (n_atoms, 1, 1))

    # (nbrs,xyz,n_entries)
    OB_vector = bond_vector[:, :, atom_index_data[:, 1]]

    neighbor_mask = neighbor_mask[atom_index_data[:, 1], :]
    # We don't want to get angle A--O--A. So, we remove A from the neighbor list of O
    neighbor_mask[np.arange(len(neighbor_mask)), atom_index_data[:, 0]] = False

    # (n_entries,nbrs)
    angle = np.sum(AO_vector * OB_vector, axis=1).T
    angle = angle * np.where(neighbor_mask, 1, np.nan)

    avg_angle = np.nanmean(angle, axis=1)
    var_angle = np.nanvar(angle, axis=1)
    min_angle = np.nanmin(angle, axis=1)
    max_angle = np.nanmax(angle, axis=1)

    return np.vstack([avg_angle, var_angle, min_angle, max_angle]).T


def get_angle_based_features(structures_df, nbr_df, edges_df):
    output = []
    for nbr_distance in [1, 2, 3]:
        one_nbr_df = nbr_df[nbr_df['nbr_distance'] == nbr_distance].copy()
        df = _angle_based_stats(structures_df, one_nbr_df, edges_df)
        df.columns = [f'{nbr_distance}_nbr_based_{c}' for c in df.columns]
        output.append(df)

    output_df = pd.concat(output, axis=1)
    return output_df


def _angle_based_stats(structures_df, nbr_df, edges_df):
    """
    both dfs should have mol_id
    both dfs should have exactly same set of mol id.
    nbr_df should have two columns: atom_index, nbr_atom_index
    """

    relevant_mol_ids = set(nbr_df.mol_id.values)
    assert relevant_mol_ids.issubset(set(structures_df.mol_id.values))
    assert relevant_mol_ids.issubset(set(edges_df.mol_id.values))

    structures_df = structures_df[structures_df.mol_id.isin(relevant_mol_ids)].copy()
    edges_df = edges_df[edges_df.mol_id.isin(relevant_mol_ids)].copy()

    nbr_df.sort_values('mol_id', inplace=True)
    structures_df.sort_values(['mol_id', 'atom_index'], inplace=True)
    edges_df.sort_values(['mol_id'], inplace=True)

    # which all atoms are neighbors of the relevant atom.
    neighbor_mask = compute_neighbor_mask(edges_df)

    atom_index_start_df = nbr_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).to_frame('nbr')
    xyz_mol_start_df = structures_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).to_frame('structure')
    assert atom_index_start_df.index.equals(xyz_mol_start_df.index)
    mol_start_idx = pd.concat([atom_index_start_df, xyz_mol_start_df], axis=1).values.astype(int)

    xyz_data = structures_df[['x', 'y', 'z']].values
    atom_index_data = nbr_df[['atom_index', 'nbr_atom_index']].values

    angle_features = np.zeros((nbr_df.shape[0], 4))
    for i, start_idx in enumerate(tqdm_notebook(mol_start_idx)):
        atom_index_start_idx = start_idx[0]
        atom_index_end_idx = mol_start_idx[i + 1][0] if i + 1 < len(mol_start_idx) else atom_index_data.shape[0]

        xyz_start_idx = start_idx[1]
        xyz_end_idx = mol_start_idx[i + 1][1] if i + 1 < len(mol_start_idx) else xyz_data.shape[0]
        angle_features[atom_index_start_idx:atom_index_end_idx] = compute_angle_one_molecule(
            xyz_data[xyz_start_idx:xyz_end_idx],
            atom_index_data[atom_index_start_idx:atom_index_end_idx],
            neighbor_mask[i],
        )

    angle_feature_df = pd.DataFrame(angle_features, index=nbr_df.index, columns=['avg', 'var', 'min', 'max'])
    angle_feature_df['mol_id'] = nbr_df['mol_id']
    angle_feature_df['atom_index'] = nbr_df['atom_index']

    output_df = angle_feature_df.groupby(['mol_id', 'atom_index']).mean()
    output_df.columns = [f'angle_{c}' for c in output_df.columns]
    return output_df
