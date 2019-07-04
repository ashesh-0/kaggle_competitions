import pandas as pd
import numpy as np
from bond_features import get_bond_data
from tqdm import tqdm_notebook

# def get_index_for_false_H_bond(atom_data, atom_col, atom_idx_col, dis, molecule_edges, H_restrict_dict):
#     if atom_data[atom_col] == 'H':
#         atom_index = atom_data[atom_idx_col]
#         if atom_index in H_restrict_dict:
#             prev_bond = H_restrict_dict[atom_index]
#             if prev_bond[0] <= dis:
#                 return len(molecule_edges)
#             else:
#                 H_restrict_dict[atom_index] = [dis, len(molecule_edges)]
#                 return prev_bond[1]
#         else:
#             H_restrict_dict[atom_index] = [dis, len(molecule_edges)]

#     return -1


def find_edges(structures_df, bond_len_factor=1.1):
    bonds_df = get_bond_data(return_limited=False)[['atom_0', 'atom_1', 'standard_bond_length']]
    bonds_df = bonds_df.groupby(['atom_0', 'atom_1']).max().sort_index()

    structures_df = structures_df.sort_values('molecule_name').reset_index()
    structures_df.index.name = 'id'

    structures_df = structures_df[['molecule_name', 'atom', 'atom_index', 'x', 'y', 'z']]
    structures_values = structures_df.values
    at_col = 1
    ati_col = 2
    x_col = 3
    y_col = 4
    z_col = 5

    temp_df = structures_df.reset_index()
    start_end = temp_df.groupby('molecule_name').agg({'id': ['first', 'last']})['id'].reset_index()
    start_end = start_end[['molecule_name', 'first', 'last']].values
    output = []
    for row in tqdm_notebook(start_end):
        molecule_edges = []
        # false_edge_indices = []
        # H_restrict_dict = {}
        mn = row[0]
        s = row[1]
        e = row[2]

        for l_idx in range(s, e):
            for r_idx in range(l_idx + 1, e + 1):
                la = structures_values[l_idx]
                ra = structures_values[r_idx]
                l_cor = la[[x_col, y_col, z_col]]
                r_cor = ra[[x_col, y_col, z_col]]
                dis = np.sqrt(np.power((l_cor - r_cor), 2).sum())
                if dis <= bonds_df.at[(la[at_col], ra[at_col]), 'standard_bond_length'] * bond_len_factor:
                    # l_del_idx = get_index_for_false_H_bond(la, at_col, ati_col, dis, molecule_edges, H_restrict_dict)
                    # r_del_idx = get_index_for_false_H_bond(ra, at_col, ati_col, dis, molecule_edges, H_restrict_dict)

                    # if l_del_idx == len(molecule_edges) or r_del_idx == len(molecule_edges):
                    #     print('continuing')
                    #     continue
                    # if l_del_idx != -1:
                    #     false_edge_indices.append(l_del_idx)
                    # if r_del_idx != -1:
                    #     false_edge_indices.append(r_del_idx)

                    molecule_edges.append([mn, la[ati_col], ra[ati_col], dis])

        # output += np.delete(np.array(molecule_edges), false_edge_indices, axis=0).tolist()
        output += molecule_edges

    edge_df = pd.DataFrame(output, columns=['molecule_name', 'atom_index_0', 'atom_index_1', 'distance'])
    return edge_df
