"""
We try to detect bond related features.
Molecule level features
- Number of bonds
- single, double,triple bond

left atom - right atom features.
Received idea from https://www.kaggle.com/adrianoavelar/eachtype
"""
import pandas as pd
import numpy as np
from common_utils_molecule_properties import find_distance_btw_point
from common_data_molecule_properties import get_bond_data


def add_bond_features(X_df):
    """
    Using the information about standard C-H,N-H and O-H bond distances, we compute features
    which aims to find a notion of deviation from the expected bond distances.
    """
    assert len(set(['atom_1', 'atom_0', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(X_df.columns)) == 0
    bond_df = get_bond_data()[['atom_0', 'atom_1', 'standard_bond_length', 'standard_bond_energy']]
    bond_df = bond_df.groupby(['atom_0', 'atom_1']).mean().reset_index()

    X_df.reset_index(inplace=True)
    X_df = pd.merge(X_df, bond_df, how='left', on=['atom_0', 'atom_1'])
    dis = find_distance_btw_point(X_df, 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1').astype(np.float32)
    X_df['BF_frac_bond_len'] = dis / X_df['standard_bond_length']
    X_df['BF_diff_bond_len'] = dis - X_df['standard_bond_length']

    X_df['BF_frac_bond_energy_2'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(2)
    X_df['BF_frac_bond_energy_4'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(4)
    X_df['BF_frac_bond_energy_6'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(6)
    X_df['BF_frac_bond_energy'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len']

    X_df.drop(['standard_bond_length', 'standard_bond_energy'], axis=1, inplace=True)

    X_df.set_index('id', inplace=True)
    return X_df
