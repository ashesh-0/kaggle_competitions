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


def bfs(nodes, parents_dict, target, edges, molecule_name, depth, max_depth):
    """
    stack is filled with the sequence of nodes from atom_index_0 to atom_index_1
    """
    # cycle
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


# def dfs(node, stack, target, edges, molecule_name, max_depth):
#     """
#     stack is filled with the sequence of nodes from atom_index_0 to atom_index_1
#     """
#     # print(node, stack)
#     # cycle
#     if node in stack:
#         return False

#     if len(stack) >= max_depth:
#         return False

#     stack.append(node)

#     if node == target:
#         return True

#     key = (molecule_name, node)
#     if key not in edges:
#         # print('Absent key', key)
#         stack.pop()
#         return False

#     for child in edges[key]:
#         output = dfs(child, stack, target, edges, molecule_name, max_depth)
#         if output is True:
#             return output

#     stack.pop()
#     return False


def get_intermediate_atoms_link(edges_df, X_df, max_path_len=10):
    """
    In the molecule structure, what atoms come in from atom_index_0 to atom_index_1 (in order)
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
