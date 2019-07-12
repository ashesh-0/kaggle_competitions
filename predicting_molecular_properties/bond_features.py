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
from common_utils import find_distance_btw_point


def add_bond_features(X_df):
    """
    Using the information about standard C-H,N-H and O-H bond distances, we compute features
    which aims to find a notion of deviation from the expected bond distances.
    """
    assert len(set(['atom_1', 'atom_0', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(X_df.columns)) == 0
    bond_df = get_bond_data().drop('bond_type', axis=1)

    X_df.reset_index(inplace=True)
    X_df = pd.merge(X_df, bond_df, how='left', on=['atom_0', 'atom_1'])
    dis = find_distance_btw_point(X_df, 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1').astype(np.float32)
    X_df['BF_frac_bond_len'] = dis / X_df['standard_bond_length']
    X_df['BF_diff_bond_len'] = dis - X_df['standard_bond_length']

    X_df['BF_frac_bond_energy_2'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(2)
    X_df['BF_frac_bond_energy_3'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(3)
    X_df['BF_frac_bond_energy_4'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(4)
    X_df['BF_frac_bond_energy'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len']

    X_df.drop(['standard_bond_length', 'standard_bond_energy'], axis=1, inplace=True)

    X_df.set_index('id', inplace=True)
    return X_df


def get_electonegativity():
    df = pd.Series(index=['H', 'O', 'C', 'N', 'F'], dtype=np.float16)
    val_dict = {
        'H': 2.1,
        'O': 3.44,
        'C': 2.55,
        'N': 3.04,
        'F': 3.98,
    }
    for key, value in val_dict.items():
        df.loc[key] = value
    df.name = 'Electronegativity'
    df.index.name = 'atom'
    return df


def get_bond_data(return_limited=True):
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies
    # http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
    # CH
    # HH
    # NH
    df = pd.DataFrame(
        [
            # H bonds
            ['H', 'C', 0, 1.09, 413],
            ['H', 'H', 0, 0.74, 436],
            ['H', 'N', 0, 1.01, 391],
            ['H', 'O', 0, 0.96, 459],
            ['H', 'F', 0, 0.92, 565],
            # C-N bonds
            ['C', 'N', 0, 1.47, 308],
            ['C', 'N', 1, 1.35, 450],
            ['C', 'N', 2, 1.27, 615],
            ['C', 'N', 3, 1.16, 887],
            ['C', 'F', 0, 1.35, 205],
            # C-O bonds
            ['C', 'O', 0, 1.40, 360],
            ['C', 'O', 1, 1.23, 799],
            ['C', 'O', 2, 1.14, 1072],
            # C-C bonds
            ['C', 'C', 0, 1.54, 348],
            ['C', 'C', 1, 1.34, 614],
            ['C', 'C', 0, 1.20, 839],
            # N-O
            ['N', 'O', 0, 1.4, 201],
            ['N', 'O', 1, 1.21, 607],
            # ['N', 'N', 0, 1.45, 167],
            ['N', 'N', 0, 1.25, 167],
            ['N', 'N', 0, 1.10, 167],
            ['N', 'F', 0, 1.36, 283],
            # o-o
            ['O', 'O', 0, 1.48, 142],
            ['O', 'O', 1, 1.21, 494],
            ['O', 'F', 0, 1.42, 190],
            # F
            ['F', 'F', 0, 1.42, 155],
        ],
        columns=['atom_0', 'atom_1', 'bond_type', 'standard_bond_length', 'standard_bond_energy'],
    )
    if return_limited:
        df = df.iloc[:3].copy()

    df[['standard_bond_energy', 'standard_bond_length']] = df[['standard_bond_energy', 'standard_bond_length']].astype(
        np.float32)
    df2 = df[df.atom_0 != df.atom_1].copy()
    atom_1 = df2.atom_1.copy()
    df2['atom_1'] = df2.atom_0
    df2['atom_0'] = atom_1
    output_df = pd.concat([df, df2], ignore_index=True)
    # add electronegativity data.
    elec_df = get_electonegativity().reset_index()
    output_df = pd.merge(output_df, elec_df, how='left', left_on=['atom_0'], right_on=['atom'])
    output_df.rename({'Electronegativity': 'atom_0_Electronegativity'}, inplace=True, axis=1)
    output_df.drop(['atom'], inplace=True, axis=1)

    output_df = pd.merge(output_df, elec_df, how='left', left_on=['atom_1'], right_on=['atom'])
    output_df.rename({'Electronegativity': 'atom_1_Electronegativity'}, inplace=True, axis=1)
    output_df.drop(['atom'], inplace=True, axis=1)
    output_df['Electronegativity_diff'] = output_df['atom_1_Electronegativity'] - output_df['atom_0_Electronegativity']
    return output_df


# if __name__ == '__main__':
#     from sklearn.preprocessing import LabelEncoder
#     from common_utils import get_structure_data
#     DIR = '/home/ashesh/Documents/initiatives/kaggle_competitions/predicting_molecular_properties/data/'
#     X_df = pd.read_csv(DIR + 'train.csv', index_col=0)
#     structures_df = pd.read_csv(DIR + 'structures.csv')

#     lb = LabelEncoder()
#     structures_df['atom'] = lb.fit_transform(structures_df.atom)
# df = unsaturation_count_features(structures_df, lb)
# print(df.head())

# X_df = get_structure_data(X_df, structures_df)
# df = bond_features(X_df, lb)
# print(df.head().T)
