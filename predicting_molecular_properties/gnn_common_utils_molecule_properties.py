import numpy as np
from tqdm import tqdm_notebook


def _permute_one_column(X, column_index):
    shape = X.shape
    X = X.copy()
    X_reshaped = X.reshape((-1, shape[-1]))
    X_reshaped[:, column_index] = np.random.permutation(X_reshaped[:, column_index])
    X_permuted = X_reshaped.reshape(shape)
    return X_permuted


def permutation_importance(model, X_edges, X_atoms, y, metric, verbose=True):
    """
    Permutes the features. If performance doesn't change a lot then it is useless.
    """
    # Taken from here https://www.kaggle.com/speedwagon/permutation-importance
    edges_results = {}
    atom_results = {}
    preds = model.predict({'adj_input': X_edges, 'nod_input': X_atoms})

    edges_results['base_score'] = metric(y, preds)
    atom_results['base_score'] = metric(y, preds)
    if verbose:
        print(f'Base score {edges_results["base_score"]:.5}')

    for col_index in tqdm_notebook(list(range(X_edges.shape[-1]))):
        X_edges_permuted = _permute_one_column(X_edges, col_index)
        # Some checks
        same = np.all(np.all(X_edges_permuted[:100] == X_edges[:100], axis=0), axis=0)
        if same[col_index]:
            print('Permutation did not change for:', col_index)

        assert np.sum(same) >= X_edges.shape[-1] - 1

        preds = model.predict({'adj_input': X_edges_permuted, 'nod_input': X_atoms})
        del X_edges_permuted

        edges_results[col_index] = metric(y, preds)
        if verbose:
            print(f'EDGES:column: {col_index} - {edges_results[col_index]:.5}')

    for col_index in tqdm_notebook(list(range(X_atoms.shape[-1]))):
        X_atoms_permuted = _permute_one_column(X_atoms, col_index)
        # Some checks
        same = np.all(np.all(X_atoms_permuted[:100] == X_atoms[:100], axis=0), axis=0)
        if same[col_index]:
            print('Permutation did not change for:', col_index)

        assert np.sum(same) == X_atoms.shape[-1] - 1

        preds = model.predict({'adj_input': X_edges, 'nod_input': X_atoms_permuted})
        del X_atoms_permuted

        atom_results[col_index] = metric(y, preds)
        if verbose:
            print(f'ATOMS:column: {col_index} - {atom_results[col_index]:.5}')

    return {'edge': edges_results, 'atom': atom_results}


def compute_diff_vector_one_molecule(data):
    """
    Each row corresponds to one molecule.
    each column coresponds to one feature.
    We want to compute difference of that feature between all molecules.
    """
    n_atoms = data.shape[0]
    A = np.tile(data.T, (n_atoms, 1, 1)).T
    diff_vector = A - A.T
    return diff_vector


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


def set_neighbor_one_molecule(index_data, output_array):
    """
    output array is 29*29 matrix. In that we need to fill
    """
    output_array[:, :] = False
    output_array[index_data[:, 0], index_data[:, 1]] = True
    output_array[index_data[:, 1], index_data[:, 0]] = True


def compute_neighbor_mask(edges_df):
    """
    Returns (#molecule_count,max_atom_count,max_atom_count) boolean array containing True for neighbor indices
    """
    atom_cnt = 29
    edges_df.sort_values('mol_id', inplace=True)
    start_idx = edges_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).astype(int).values
    mol_cnt = start_idx.shape[0]

    data = np.zeros((mol_cnt, atom_cnt, atom_cnt), dtype=bool)
    index_data = edges_df[['atom_index_0', 'atom_index_1']].values
    for i, start_index in enumerate(start_idx):
        end_index = start_idx[i + 1] if i + 1 < len(start_idx) else edges_df.shape[0]
        set_neighbor_one_molecule(index_data[start_index:end_index], data[i])
    return data


def eval_metric(actual, predic):
    actual = actual.reshape(-1)
    predic = predic.reshape(-1)
    filtr = actual != 0
    error = np.mean(np.abs(actual[filtr] - predic[filtr]))
    return np.log(error)
