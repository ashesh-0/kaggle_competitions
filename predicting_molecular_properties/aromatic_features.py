import numpy as np
import pandas as pd
from common_data_molecule_properties import get_electonegativity
from tqdm import tqdm_notebook


def _get_start_end_index(index_data, idx, size):
    start_index = index_data[idx]
    end_index = index_data[idx + 1] if idx + 1 < len(index_data) else size
    return start_index, end_index


def compute_aromatic_features(structures_df, cycles_df, edges_df, aromatic_df):
    assert 'mol_id' in structures_df
    assert 'mol_id' in cycles_df
    assert 'mol_id' in edges_df
    assert 'mol_id' in aromatic_df
    # make sure each as same mol_ids.
    relevant_mol_ids = aromatic_df.mol_id.unique()
    structures_df = structures_df[structures_df.mol_id.isin(relevant_mol_ids)].copy()
    cycles_df = cycles_df[cycles_df.mol_id.isin(relevant_mol_ids)].copy()
    edges_df = edges_df[edges_df.mol_id.isin(relevant_mol_ids)].copy()

    # checks to ensure same set of relevant_mol_ids exist.
    assert set(structures_df.mol_id.unique()) == set(relevant_mol_ids)
    assert set(cycles_df.mol_id.unique()) == set(relevant_mol_ids)
    assert set(edges_df.mol_id.unique()) == set(relevant_mol_ids)

    en_dict = get_electonegativity().to_dict()

    lone_pair_dict = {'C': 0, 'H': 0, 'N': 1, 'O': 1, 'F': 1}
    structures_df['EN'] = structures_df['atom'].map(en_dict)
    structures_df['LonePair'] = structures_df['atom'].map(lone_pair_dict)
    structures_df['LonePair'] = structures_df['LonePair'] / np.sqrt(structures_df['EN'])
    # We don't want these to get summed up. Unnecessary noise.
    structures_df.loc[structures_df.atom.isin(['C,H']), 'EN'] = 0

    structures_df.sort_values(['mol_id', 'atom_index'], inplace=True)
    cycles_df.sort_values('mol_id', inplace=True)
    edges_df.sort_values('mol_id', inplace=True)
    aromatic_df.sort_values('mol_id', inplace=True)

    start_structures_df = structures_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int)
    mol_ids = start_structures_df.index.tolist()

    start_structures_df_indices = start_structures_df.values
    structures_data = structures_df[['EN', 'LonePair']].values

    start_cycles_df_indices = cycles_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    cycles_data = cycles_df[[str(i) for i in range(10)]].astype(int).values

    start_edges_df_indices = edges_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    edges_data = edges_df[['atom_index_0', 'atom_index_1']].astype(int).values

    start_aromatic_df_indices = aromatic_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    aromatic_data = aromatic_df[['atom_index_0', 'atom_index_1']].astype(int).values

    molecule_features = np.zeros((len(mol_ids), 4))
    atom_features = np.zeros((structures_data.shape[0], 2))
    for idx, mol_id in enumerate(tqdm_notebook(mol_ids)):
        s_start_index, s_end_index = _get_start_end_index(start_structures_df_indices, idx, structures_data.shape[0])
        c_start_index, c_end_index = _get_start_end_index(start_cycles_df_indices, idx, cycles_data.shape[0])
        e_start_index, e_end_index = _get_start_end_index(start_edges_df_indices, idx, edges_data.shape[0])
        a_start_index, a_end_index = _get_start_end_index(start_aromatic_df_indices, idx, aromatic_data.shape[0])

        (molecule_features_one_mol, atom_features_one_mol) = compute_aromatic_feature_one_molecule(
            structures_data[s_start_index:s_end_index],
            cycles_data[c_start_index:c_end_index],
            edges_data[e_start_index:e_end_index],
            aromatic_data[a_start_index:a_end_index],
        )
        molecule_features[idx, :] = molecule_features_one_mol
        atom_features[s_start_index:s_end_index, :] = atom_features_one_mol

    molecule_feature_df = pd.DataFrame(
        molecule_features,
        index=mol_ids,
        columns=['InCycle_EN', 'InCycle_LonePair', 'NbrCycle_EN', 'NbrCycle_LonePair'])
    molecule_feature_df.index.name = 'mol_id'

    atom_features_df = pd.DataFrame(
        atom_features, index=structures_df.index, columns=['is_aromatic_atom', 'is_aromatic_nbr'])

    atom_features_df['mol_id'] = structures_df['mol_id']
    atom_features_df['atom_index'] = structures_df['atom_index']

    molecule_feature_df.drop('InCycle_LonePair', axis=1, inplace=True)
    return atom_features_df.join(molecule_feature_df, how='outer', on='mol_id')


def compute_aromatic_feature_one_molecule(structures_data, cycle_nodes, edge_data, aromatic_data):
    """
    structures_data: 2 columns: EN,LonePair
    cycle_nodes: list of cycles. each row, one cycle.columns have atom_index
    edge_data: 2 columns, atom_index_0,atom_index_1
    aromatic_data: 2 columns, atom_index_0,atom_index_1. Only those rows which form an aromatic bond.
    """
    n_atoms = structures_data.shape[0]
    adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
    adjacency_matrix[edge_data[:, 0], edge_data[:, 1]] = True
    adjacency_matrix[edge_data[:, 1], edge_data[:, 0]] = True

    aromatic_edges_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
    aromatic_edges_matrix[aromatic_data[:, 0], aromatic_data[:, 1]] = True
    aromatic_edges_matrix[aromatic_data[:, 1], aromatic_data[:, 0]] = True

    # 1st dimension is replica.,3rd is features.
    structures_data_matrix = np.tile(structures_data, (n_atoms, 1, 1))

    all_aromatic_cycles = []
    for one_cycle in cycle_nodes:
        one_cycle = one_cycle[one_cycle != -1]

        # first and last node needs to be checked for connection
        one_cycle = np.append(one_cycle, one_cycle[0])

        edge_left_nodes = one_cycle[:-1]
        edge_right_nodes = one_cycle[1:]
        if not np.all(aromatic_edges_matrix[edge_left_nodes, edge_right_nodes]):
            continue
        # we have an aromatic cycle
        one_cycle = one_cycle[:-1]
        all_aromatic_cycles.append(one_cycle)

    cycle_atoms = list(set(np.concatenate(all_aromatic_cycles)))
    # Is this atom aromatic.
    is_aromatic_atom_feature = np.zeros((n_atoms, 1), dtype=bool)
    is_aromatic_atom_feature[cycle_atoms] = True

    # Is any of its nbr aromatic.
    is_aromatic_nbr_feature = (aromatic_edges_matrix * adjacency_matrix).sum(axis=1).reshape(-1, 1)

    # -1 is atom_type
    in_cycle_features = structures_data[cycle_atoms, :].sum(axis=0)

    # We want to ignore in_cycle edges
    adjacency_matrix[aromatic_edges_matrix] = False

    # We take all the neighbors of aromatic atoms and find EN and lone pair sum.
    nbr_cycle_features = (adjacency_matrix.reshape(n_atoms, n_atoms, 1)[cycle_atoms] *
                          structures_data_matrix[cycle_atoms, :, :]).sum(axis=1).sum(axis=0)

    # same for all molecule
    molecule_features = np.concatenate([in_cycle_features, nbr_cycle_features])
    atom_features = np.hstack([is_aromatic_atom_feature, is_aromatic_nbr_feature])
    return (molecule_features, atom_features)
