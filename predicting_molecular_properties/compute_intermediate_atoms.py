from tqdm import tqdm_notebook
import pandas as pd
import numpy as np


def _get_path_to_src(end_node, parents_dict):
    output = []
    src = end_node
    while src != -1:
        output.append(src)
        src = parents_dict[src]

    return list(reversed(output))


def bfs_for_neighbors(nodes, parents_dict, edges, molecule_name, kneighbor_dict, depth, max_depth):
    """
    stack is filled with the sequence of nodes from atom_index_0 to atom_index_1
    """
    # We store neighbors
    if depth > 0:
        kneighbor_dict[depth] = nodes

    if depth >= max_depth:
        return

    next_layer = []
    for parent_node in nodes:

        key = (molecule_name, parent_node)
        if key not in edges:
            # print('Absent key', key)
            continue

        for child in edges[key]:
            if child in parents_dict or child in next_layer:
                continue

            next_layer.append(child)
            parents_dict[child] = parent_node

    return bfs_for_neighbors(next_layer, parents_dict, edges, molecule_name, kneighbor_dict, depth + 1, max_depth)


def bfs(nodes, parents_dict, target, edges, molecule_name, depth, max_depth):
    """
    stack is filled with the sequence of nodes from atom_index_0 to atom_index_1
    """

    if depth >= max_depth:
        return []

    next_layer = []
    for parent_node in nodes:

        if parent_node == target:
            return _get_path_to_src(target, parents_dict)

        key = (molecule_name, parent_node)
        if key not in edges:
            # print('Absent key', key)
            continue

        for child in edges[key]:
            if child in parents_dict or child in next_layer:
                continue

            next_layer.append(child)
            parents_dict[child] = parent_node

    return bfs(next_layer, parents_dict, target, edges, molecule_name, depth + 1, max_depth)


def get_neighbor_atoms(edges_df, X_df, max_path_len=5):
    """
        Returns Neighbor atom_indices for each atom_index present in X_df
    """
    # assuming symmetric edges and atom information.
    # id to column
    X_df = X_df.reset_index()

    edges = edges_df.groupby(['molecule_name', 'atom_index_0'])['atom_index_1'].apply(list).to_dict()
    data1 = X_df[['id', 'molecule_name', 'atom_index_0',
                  'atom_index_1']].groupby(['molecule_name', 'atom_index_0']).first().reset_index()
    data1.rename({'atom_index_0': 'atom_index'}, axis=1, inplace=True)
    data1.drop('atom_index_1', axis=1, inplace=True)

    data2 = X_df[['id', 'molecule_name', 'atom_index_0',
                  'atom_index_1']].groupby(['molecule_name', 'atom_index_1']).first().reset_index()
    data2.rename({'atom_index_1': 'atom_index'}, axis=1, inplace=True)
    data2.drop('atom_index_0', axis=1, inplace=True)

    data = pd.concat([data1, data2])
    data = data.groupby(['molecule_name', 'atom_index']).first().reset_index()
    data = data[['id', 'molecule_name', 'atom_index']].values
    kneighbor_output = []
    for row in tqdm_notebook(data):
        idx = row[0]
        mn = row[1]
        s = row[2]
        parents_dict = {s: -1}
        kneighbor_dict = {}
        bfs_for_neighbors([s], parents_dict, edges, mn, kneighbor_dict, 0, max_path_len)

        nbr_depth = []
        nbr_ai = []
        for dep, kneighbors in kneighbor_dict.items():
            nbr_depth += [dep] * len(kneighbors)
            nbr_ai += kneighbors

        # id,atom_index,neighbor_bond_distance,atom_index
        kneighbor_output += [np.vstack([[idx] * len(nbr_ai), [s] * len(nbr_ai), nbr_depth, nbr_ai]).T]

    kneighbor_df = pd.DataFrame(
        np.concatenate(kneighbor_output),
        columns=['id', 'atom_index', 'nbr_distance', 'nbr_atom_index'],
        dtype=np.int32)

    kneighbor_df[['atom_index', 'nbr_distance',
                  'nbr_atom_index']] = kneighbor_df[['atom_index', 'nbr_distance', 'nbr_atom_index']].astype(np.uint8)

    return kneighbor_df


def get_intermediate_atoms_link(edges_df, X_df, max_path_len=10):
    """
    Returns In the molecule structure, what atoms come in from atom_index_0 to atom_index_1 (in order)
    """
    # assuming symmetric edges and atom information.
    edges = edges_df.groupby(['molecule_name', 'atom_index_0'])['atom_index_1'].apply(list).to_dict()
    data = X_df[['molecule_name', 'atom_index_0', 'atom_index_1']].values
    output = [[]] * len(data)
    # max path_len -2 atoms between s and e

    for idx, row in tqdm_notebook(enumerate(data)):
        mn = row[0]
        s = row[1]
        e = row[2]
        parents_dict = {s: -1}
        path = bfs([s], parents_dict, e, edges, mn, 0, max_path_len)

        output[idx] = path + [-1] * (max_path_len - len(path))

    return pd.DataFrame(output, index=X_df.index, dtype=np.int16)
