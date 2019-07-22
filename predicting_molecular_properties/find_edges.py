import pandas as pd
import numpy as np
from bond_features import get_bond_data
from tqdm import tqdm


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
    for row in tqdm(start_end):
        molecule_edges = []
        mn = row[0]
        s = row[1]
        e = row[2]

        for l_idx in range(s, e):
            for r_idx in range(l_idx + 1, e + 1):
                la = structures_values[l_idx]
                ra = structures_values[r_idx]
                l_cor = la[[x_col, y_col, z_col]]
                r_cor = ra[[x_col, y_col, z_col]]
                bnd_vector = r_cor - l_cor
                dis = np.sqrt(np.power(bnd_vector, 2).sum())
                bnd_vector = bnd_vector / dis
                if dis <= bonds_df.at[(la[at_col], ra[at_col]), 'standard_bond_length'] * bond_len_factor:
                    molecule_edges.append([mn, la[ati_col], ra[ati_col], dis] + bnd_vector.tolist())

        output += molecule_edges

    edge_df = pd.DataFrame(output, columns=['molecule_name', 'atom_index_0', 'atom_index_1', 'distance', 'x', 'y', 'z'])
    edge_df[['atom_index_0', 'atom_index_1']] = edge_df[['atom_index_0', 'atom_index_1']].astype(np.uint8)
    edge_df[['distance', 'x', 'y', 'z']] = edge_df[['distance', 'x', 'y', 'z']].astype(np.float32)

    return edge_df
