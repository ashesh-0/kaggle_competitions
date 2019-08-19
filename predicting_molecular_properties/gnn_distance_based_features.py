from gnn_common_utils_molecule_properties import (compute_bond_vector_one_molecule, compute_neighbor_mask,
                                                  compute_diff_vector_one_molecule)

import pandas as pd
from common_data_molecule_properties import get_electonegativity
import numpy as np


def induced_electronegativity_atom_feature(structures_df, edges_df):
    """
    Assume edges_df are sorted by mol_id and structures_df is sorted by mol_id and atom_index.
    """

    mol_ids = structures_df.mol_id.unique()
    assert set(mol_ids) == set(edges_df.mol_id.unique())
    mask = compute_neighbor_mask(edges_df)

    structures_df['EN'] = structures_df['atom'].map(get_electonegativity())
    start_idx = structures_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    str_data = structures_df[['EN', 'x', 'y', 'z']].values
    data = np.zeros((structures_df.shape[0], 2))
    for i, start_index in enumerate(start_idx):
        end_index = start_idx[i + 1] if i + 1 < len(start_idx) else structures_df.shape[0]
        str_data_one_mol = str_data[start_index:end_index, :]
        n_atoms = str_data_one_mol.shape[0]
        atoms_EN_diff = compute_diff_vector_one_molecule(str_data_one_mol[:, :1]).reshape((n_atoms, n_atoms))

        mask_one_mol = mask[i][:n_atoms, :n_atoms]
        # only neighbors
        atoms_EN_diff = atoms_EN_diff * mask_one_mol
        # normalize by distance
        bond_data = compute_bond_vector_one_molecule(str_data_one_mol[:, 1:])
        bond_len = bond_data['bond_length']
        bond_vec = bond_data['bond_vector']

        atoms_EN_diff = atoms_EN_diff / bond_len
        EN_vector = np.sum(bond_vec * atoms_EN_diff.reshape((n_atoms, 1, n_atoms)), axis=2)

        data[start_index:end_index, 0] = atoms_EN_diff.sum(axis=1)
        data[start_index:end_index, 1] = np.linalg.norm(EN_vector, axis=1)
    output_df = pd.DataFrame(data, index=structures_df.index, columns=['EN_diff_sum', 'EN_vec_sum'])
    output_df['atom_index'] = structures_df['atom_index']
    output_df['mol_id'] = structures_df['mol_id']
    return output_df
