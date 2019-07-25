import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from decorators import timer


def get_cycle_features_dict(cycles_df):
    filtr = cycles_df != -1
    cycle_len = filtr.sum(axis=1).groupby('molecule_name').mean().to_frame('cycle_length')
    cycle_nodes_df = cycles_df[filtr].stack().astype(np.int16).groupby('molecule_name').agg([list])
    return pd.concat([cycle_len, cycle_nodes_df], axis=1).to_dict()


def get_ia_dict(ia_df):
    ia_df = ia_df.copy()
    ia_df.columns.name = 'nbr_position'
    temp = ia_df[ia_df != -1].stack().reset_index().groupby(['id'])
    feat_df = temp.nbr_position.agg([list, len])
    return feat_df.to_dict()


@timer('CycleFeatures')
def add_cycle_features(cycles_df, ia_df, X_df):
    """
    #atom_index_0 in cycle.
    #atom_index_1 in cycle.
    # atom_index_0_neighbor in cycle
    # atom_index_1_neighbor in cycle.
    # cycle_length
    """
    cycle_features_dict = get_cycle_features_dict(cycles_df)
    id_to_molecule_name = X_df['molecule_name'].to_dict()

    ia_values = ia_df.reset_index().values
    features = np.zeros((len(ia_values), 5))
    # very big cycle ~ no cycle.
    features[:, 4] = 100

    idx = -1
    for row in tqdm_notebook(ia_values):
        idx += 1
        mn = id_to_molecule_name[row[0]]
        ai_0_idx = 1
        ai_1_idx = row.tolist().index(-1) - 1
        cycle = cycle_features_dict['list'].get(mn, None)
        if cycle is None:
            continue

        cycle_len = cycle_features_dict['cycle_length'][mn]
        feature = [
            row[ai_0_idx] in cycle,
            row[ai_1_idx] in cycle,
            row[ai_0_idx + 1] in cycle,
            row[ai_1_idx - 1] in cycle,
            cycle_len,
        ]
        features[idx, :] = feature

    output_df = pd.DataFrame(
        features,
        index=ia_df.index,
        columns=['in_cycle_atom_0', 'in_cycle_atom_1', 'in_cycle_atom_0_nbr', 'in_cycle_atom_1_nbr', 'cycle_len'],
        dtype=np.float16)
    output_df[['in_cycle_atom_0', 'in_cycle_atom_1', 'in_cycle_atom_0_nbr', 'in_cycle_atom_1_nbr']] = output_df[[
        'in_cycle_atom_0', 'in_cycle_atom_1', 'in_cycle_atom_0_nbr', 'in_cycle_atom_1_nbr'
    ]].astype(bool)

    for col in output_df:
        X_df[col] = output_df[col]
