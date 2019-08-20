"""
Yukawa and Coulomb potential.
"""
import numpy as np
from tqdm import tqdm_notebook
import pandas as pd

from common_utils_molecule_properties import get_structure_data, get_symmetric_edges, add_structure_data_to_edge
from common_data_molecule_properties import get_electonegativity

# from decorators import timer


def compute_potential_one_molecule(data: np.array):
    """
    Columns are x,y,z followed by various approximations of induced charge.
    """
    mol_count = data.shape[0]
    data_copy = np.tile(data.T, (mol_count, 1, 1))
    r = np.sqrt(np.square(data_copy[:, :3, :].T - data_copy[:, :3, :]).sum(axis=1))
    np.fill_diagonal(r, 1e10)
    r = r.reshape(mol_count, 1, mol_count)
    q1q2 = data_copy[:, 3:, :].T * data_copy[:, 3:, :]
    yukawa = (q1q2 * np.exp(-r) / r).sum(axis=2)
    coulomb = (q1q2 / r).sum(axis=2)
    return (yukawa, coulomb)


def compute_induced_charge_on_atoms(structures_df, edges_df):
    electroneg_df = get_electonegativity()
    structures_df = structures_df.copy()
    structures_df['EN'] = structures_df['atom'].map(electroneg_df)
    edges_df = get_symmetric_edges(edges_df)
    edges_df = add_structure_data_to_edge(edges_df, structures_df, cols_to_add=['atom', 'EN'])
    edges_df['Q'] = (edges_df['EN_0'] - edges_df['EN_1']) / (edges_df['distance'])
    edges_df['Q4'] = (edges_df['EN_0'] - edges_df['EN_1']) / (edges_df['distance']).pow(4)

    atom_EN_df = edges_df.groupby(['molecule_name', 'atom_index_0'])[['Q', 'Q4']].sum().reset_index()
    atom_EN_df.rename({'atom_index_0': 'atom_index'}, axis=1, inplace=True)

    structures_df.drop(['EN'], axis=1, inplace=True)
    structures_df = pd.merge(structures_df, atom_EN_df, on=['molecule_name', 'atom_index'], how='outer')
    return structures_df


@timer('AtomPotentials')
def add_atom_potential_features(X_df, structures_df, edges_df):
    """
    Yukawa potential and Coulomb potential.
    """
    structures_df = compute_induced_charge_on_atoms(structures_df, edges_df)
    mol_idx_df = structures_df.groupby('molecule_name').size().cumsum()
    mol_start_idx_df = mol_idx_df.shift(1).fillna(0).astype(int)
    start_indices = mol_start_idx_df.values
    data = structures_df[['x', 'y', 'z', 'Q', 'Q4']].values

    yukawa_arr = []
    coulomb_arr = []
    for i, start_index in tqdm_notebook(enumerate(start_indices)):
        end_index = start_indices[i + 1] if i + 1 < len(start_indices) else data.shape[0]
        yukawa, coulomb = compute_potential_one_molecule(data[start_index:end_index])
        yukawa_arr.append(yukawa)
        coulomb_arr.append(coulomb)

    coulomb_df = pd.DataFrame(
        np.concatenate(coulomb_arr), index=structures_df.index, columns=['Coulomb_Q', 'Coulomb_Q4'], dtype=np.float32)
    yukawa_df = pd.DataFrame(
        np.concatenate(yukawa_arr), index=structures_df.index, columns=['Yukawa_Q', 'Yukawa_Q4'], dtype=np.float32)

    for col in coulomb_df.columns:
        structures_df[col] = coulomb_df[col]

    for col in yukawa_df.columns:
        structures_df[col] = yukawa_df[col]

    potential_features = ['Q', 'Q4', 'Coulomb_Q', 'Coulomb_Q4', 'Yukawa_Q', 'Yukawa_Q4']
    X_df = get_structure_data(X_df, structures_df, cols_to_add=potential_features)
    return X_df
