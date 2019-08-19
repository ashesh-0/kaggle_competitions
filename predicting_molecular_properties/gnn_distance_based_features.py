from gnn_common_utils_molecule_properties import (compute_bond_vector_one_molecule, compute_neighbor_mask,
                                                  compute_diff_vector_one_molecule)

import pandas as pd
from common_data_molecule_properties import get_electonegativity
import numpy as np


def compute_potential_one_molecule(charge, bond_length):
    """
    charge:(n_atoms,1)
    bond_length:(n_atoms,n_atoms)
    """
    n_atoms = charge.shape[0]
    q1 = np.tile(charge.reshape(1, n_atoms), (n_atoms, 1)).T
    q2 = q1.T

    q1q2 = q1 * q2
    np.fill_diagonal(q1q2, 0)
    r = bond_length
    yukawa = (q1q2 * np.exp(-r) / r).sum(axis=1)
    coulomb = (q1q2 / r).sum(axis=1)
    return (yukawa, coulomb)


def induced_electronegativity_edge_feature(X_df, structures_df, en_vector_all_molecules):
    """
    en_vector_all_molecules: (n_mol,n_atoms,3)
    """

    structures_start_idx = structures_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    xyz_data = structures_df[['x', 'y', 'z']].values

    atom_index_values = X_df[['atom_index_0', 'atom_index_1']].astype(int).values
    start_idx = X_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values

    data = np.zeros((X_df.shape[0], 2))
    for mol_id, start_index in enumerate(start_idx):
        end_index = start_idx[mol_id + 1] if mol_id + 1 < len(start_idx) else X_df.shape[0]
        atom_index_one_mol = atom_index_values[start_index:end_index]

        structures_start_index = structures_start_idx[mol_id]
        structures_end_index = structures_start_idx[
            mol_id + 1] if mol_id + 1 < len(structures_start_idx) else structures_df.shape[0]

        bond_vector = compute_bond_vector_one_molecule(
            xyz_data[structures_start_index:structures_end_index])['bond_vector']

        bond_vector_X_rows = bond_vector[atom_index_one_mol[:, 0], :, atom_index_one_mol[:, 1]]

        # n_atoms,3
        en_one_mol = en_vector_all_molecules[mol_id]
        en_net_sq = np.power(en_one_mol, 2).sum(axis=1)

        # # for atom index 0
        # en_along_ai0 = np.sum(bond_vector_X_rows * en_one_mol[atom_index_one_mol[:, 0]], axis=1)
        # en_perp_ai0 = en_net_sq[atom_index_one_mol[:, 0]] - np.power(en_along_ai0, 2)
        # en_perp_ai0[en_perp_ai0 < 0] = 0
        # en_perp_ai0 = np.power(en_perp_ai0, 0.5)

        # for atom index 1
        en_along_ai1 = np.sum(bond_vector_X_rows * en_one_mol[atom_index_one_mol[:, 1]], axis=1)
        en_perp_ai1 = en_net_sq[atom_index_one_mol[:, 1]] - np.power(en_along_ai1, 2)
        en_perp_ai1[en_perp_ai1 < 0] = 0
        en_perp_ai1 = np.power(en_perp_ai1, 0.5)

        data[start_index:end_index, 0] = en_along_ai1
        data[start_index:end_index, 1] = en_perp_ai1

    output_df = pd.DataFrame(data, columns=['ai1_EN_along', 'ai1_EN_perp'], index=X_df.index)

    return output_df


def induced_electronegativity_feature(structures_df, edges_df, X_df):
    atom_data, en_vector_all_molecules = induced_electronegativity_atom_feature(structures_df, edges_df)
    edge_data = induced_electronegativity_edge_feature(X_df, structures_df, en_vector_all_molecules)
    return {'atom': atom_data, 'edge': edge_data}


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
    data = np.zeros((structures_df.shape[0], 4))
    en_vector_all_molecules = np.zeros((len(mol_ids), 29, 3), dtype=np.float16)
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
        en_vector_all_molecules[i, :n_atoms, :] = EN_vector
        induced_EN = atoms_EN_diff.sum(axis=1)
        data[start_index:end_index, 0] = induced_EN
        data[start_index:end_index, 1] = np.linalg.norm(EN_vector, axis=1)

        # Atom Potentials
        yukawa, coulomb = compute_potential_one_molecule(induced_EN, bond_len)
        data[start_index:end_index, 2] = yukawa
        data[start_index:end_index, 3] = coulomb

    output_df = pd.DataFrame(
        data, index=structures_df.index, columns=['EN_diff_sum', 'EN_vec_sum', 'yukawa', 'coulomb'])
    output_df['atom_index'] = structures_df['atom_index']
    output_df['mol_id'] = structures_df['mol_id']
    return output_df, en_vector_all_molecules
