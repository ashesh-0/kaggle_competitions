"""
Superfast computation of angles between any 3 pair of atoms.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


def compute_bond_vector_one_molecule(xyz_data):
    """
    xyz_data[i,0] is the x cordinate of ith atom.
    xyz_data[i,1] is the y cordinate of ith atom.
    xyz_data[i,1] is the z cordinate of ith atom.
    """
    assert xyz_data.shape[1] == 3
    n_atoms = xyz_data.shape[0]

    A = np.tile(xyz_data.T, (n_atoms, 1, 1)).T

    incoming_bond_vector = A - A.T
    bond_length = np.linalg.norm(incoming_bond_vector, axis=1)
    # Ensuring that same atom has very very large bond length.
    bond_length = np.where(bond_length == 0, 1e10, bond_length)
    # normalize
    incoming_bond_vector = incoming_bond_vector / bond_length.reshape(n_atoms, 1, n_atoms)
    return {'bond_length': bond_length, 'bond_vector': incoming_bond_vector}


def compute_angle_one_molecule(xyz_data, atom_index_data, neighbor_count=3):
    """
    A--O--B
    Each row in atom_index_data contains one A--O
    Here, We find angle AOB for each row, for each B belonging to nearest 3 neighbors.
    """
    data = compute_bond_vector_one_molecule(xyz_data)
    dist = data['bond_length']
    bond_vector = data['bond_vector']
    n_atoms = xyz_data.shape[0]
    # we don't want to select same atom as its neighbor. Hence the -1.
    neighbor_count = min(neighbor_count, n_atoms - 1)
    nbr_indices = dist.argsort()[:, :neighbor_count]

    OB_vector = np.zeros((neighbor_count, n_atoms, 3))
    for i in range(neighbor_count):
        ith_nbr = nbr_indices[:, i]
        OB_vector[i, :, :] = bond_vector[np.arange(n_atoms), :, ith_nbr]

    OB_vector = OB_vector[:, atom_index_data[:, 1], :]

    AO_vector = bond_vector[atom_index_data[:, 0], :, atom_index_data[:, 1]]
    AO_vector = np.tile(AO_vector, (neighbor_count, 1, 1))

    angle = np.sum(AO_vector * OB_vector, axis=2)

    avg_angle = np.mean(angle, axis=0)
    std_angle = np.std(angle, axis=0)
    min_angle = np.min(angle, axis=0)
    max_angle = np.max(angle, axis=0)

    return np.vstack([avg_angle, std_angle, min_angle, max_angle]).T


def angle_based_stats(structures_df, nbr_df):
    assert set(nbr_df.mol_id.values) == set(structures_df.mol_id.values)

    # nbr_df.sort_values('mol_id', inplace=True)
    structures_df.sort_values(['mol_id', 'atom_index'], inplace=True)

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
        )

    angle_feature_df = pd.DataFrame(angle_features, index=nbr_df.index, columns=['avg', 'std', 'min', 'max'])
    angle_feature_df['mol_id'] = nbr_df['mol_id']
    angle_feature_df['atom_index'] = nbr_df['atom_index']

    return angle_feature_df.groupby(['mol_id', 'atom_index']).mean().reset_index()
