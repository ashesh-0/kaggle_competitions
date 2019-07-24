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


def find_cycles(edges, molecule_name, max_depth):
    node = 0
    stack = []
    cycle_array = []
    _find_cycles(node, stack, edges, molecule_name, max_depth, cycle_array)
    cycle_sets = []
    output = []
    for cycle in cycle_array:
        if set(cycle) in cycle_sets:
            continue
        cycle_sets.append(set(cycle))
        output.append(cycle)

    return output


def _find_cycles(node, stack, edges, molecule_name, max_depth, cycle_array):
    """
    dfs is implemented to get cycles
    """
    # cycle
    if node in stack:
        idx = stack.index(node)
        cycle_array.append(stack[idx:].copy())
        return

    if len(stack) >= max_depth:
        return

    stack.append(node)

    key = (molecule_name, node)
    if key not in edges:
        # print('Absent key', key)
        stack.pop()
        return

    for child in edges[key]:
        # we don't want to use the same edge in reverse fashion
        if len(stack) > 1 and child == stack[-2]:
            continue
        _find_cycles(child, stack, edges, molecule_name, max_depth, cycle_array)

    stack.pop()
    return


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


def get_cycle_data(edges_df, structures_df):
    """
    Returns cycles present in the structure. each row corresponds to one cycle.
    """
    edges = edges_df.groupby(['molecule_name', 'atom_index_0'])['atom_index_1'].apply(list).to_dict()
    molecule_names = structures_df.molecule_name.unique()
    max_depth = 50
    # note that 9 is the maximum number of non H atoms in the problem
    max_cycle_len = 10
    output = []
    for mn in molecule_names:
        cycles = find_cycles(edges, mn, max_depth)
        for cycle in cycles:
            assert len(cycle) <= max_cycle_len
            row = [mn] + cycle + [-1] * (max_cycle_len - len(cycle))
            output.append(row)

    cols = ['molecule_name'] + list(map(str, list(range(10))))
    df = pd.DataFrame(output, columns=cols)
    df[cols[1:]] = df[cols[1:]].astype(np.int16)
    return df.set_index('molecule_name')


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
