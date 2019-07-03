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


def unsaturation_count_features(structures_df, atom_encoder):
    n_counts = structures_df.groupby(['molecule_name', 'atom']).size().unstack().fillna(0)
    nC = n_counts[atom_encoder.transform(['C'])[0]]
    nO = n_counts[atom_encoder.transform(['O'])[0]]
    nN = n_counts[atom_encoder.transform(['N'])[0]]
    nH = n_counts[atom_encoder.transform(['H'])[0]]
    unsatu_cnt = ((2 * nC + 2 - 2 * nO - nN) - nH) / 2
    unsatu_cnt = unsatu_cnt.to_frame('unsaturation_count')
    unsatu_cnt['unsaturation_fraction'] = unsatu_cnt['unsaturation_count'] / n_counts.sum(axis=1)
    # Not valid for non carbon atoms
    unsatu_cnt.loc[nC == 0, :] = -100
    return unsatu_cnt


def bond_features(X_df, atom_encoder):
    assert len(set(['atom_1', 'atom_0', 'x_0', 'x_1', 'y_0', 'y_1', 'z_0', 'z_1']) - set(X_df.columns)) == 0
    bond_df = get_bond_data().drop('bond_type',axis=1)
    bond_df['atom_0'] = atom_encoder.transform(bond_df['atom_0'])
    bond_df['atom_1'] = atom_encoder.transform(bond_df['atom_1'])
    X_df.reset_index(inplace=True)
    X_df = pd.merge(X_df, bond_df, how='left', on=['atom_0', 'atom_1'])
    dis = find_distance_btw_point(X_df, 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1').astype(np.float32)
    X_df['BF_frac_bond_len'] = dis / X_df['standard_bond_length']
    X_df['BF_frac_bond_energy_2'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len'].pow(2)
    X_df['BF_frac_bond_energy'] = X_df['standard_bond_energy'] / X_df['BF_frac_bond_len']
    X_df['BF_diff_bond_len'] = dis - X_df['standard_bond_length']
    X_df.rename(
        {
            'standard_bond_length': 'BF_standard_bond_length',
            'standard_bond_energy': 'BF_standard_bond_energy',
        },
        axis=1,
        inplace=True)
    X_df.set_index('id', inplace=True)
    return X_df


def get_bond_data():
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
            ['N', 'H', 0, 1.01, 391],
            # C-N bonds
            ['C', 'N', 0, 1.47, 308],
            ['C', 'N', 1, 1.35, 450],
            ['C', 'N', 2, 1.27, 615],
            ['C', 'N', 3, 1.16, 887],
            # C-O bonds
            ['C', 'O', 0, 1.40, 360],
            ['C', 'O', 1, 1.23, 799],
            ['C', 'O', 2, 1.14, 1072],
            # C-C bonds
            ['C', 'C', 0, 1.54, 348],
            ['C', 'C', 1, 1.34, 614],
            ['C', 'C', 0, 1.20, 839],
        ],
        columns=['atom_0', 'atom_1', 'bond_type', 'standard_bond_length', 'standard_bond_energy'],
    )
    df[['standard_bond_energy', 'standard_bond_length']] = df[['standard_bond_energy', 'standard_bond_length']].astype(
        np.float32)
    df2 = df[df.atom_0 != df.atom_1].copy()
    atom_1 = df2.atom_1
    df2['atom_1'] = df2.atom_0
    df2['atom_0'] = atom_1
    return pd.concat([df, df2], ignore_index=True)


if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    from common_utils import get_structure_data
    DIR = '/home/ashesh/Documents/initiatives/kaggle_competitions/predicting_molecular_properties/data/'
    X_df = pd.read_csv(DIR + 'train.csv', index_col=0)
    structures_df = pd.read_csv(DIR + 'structures.csv')

    lb = LabelEncoder()
    structures_df['atom'] = lb.fit_transform(structures_df.atom)
    # df = unsaturation_count_features(structures_df, lb)
    # print(df.head())

    X_df = get_structure_data(X_df, structures_df)
    df = bond_features(X_df, lb)
    print(df.head().T)
    import pdb; pdb.set_trace()
