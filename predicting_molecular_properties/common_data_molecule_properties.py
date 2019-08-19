import pandas as pd
import numpy as np


def get_lone_pair():
    """
    Pi donars increase the value
    """
    return {'C': 0, 'H': 0, 'F': 0.5, 'N': 1, 'O': 0.8}


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


def get_bond_data():
    # https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Chemical_Bonding/Fundamentals_of_Chemical_Bonding/Chemical_Bonds/Bond_Lengths_and_Energies
    # http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
    # https://en.wikipedia.org/wiki/Carbon%E2%80%93nitrogen_bond
    # CH
    # HH
    # NH
    df = pd.DataFrame(
        [
            # H bonds
            ['H', 'C', 1, 1.09, 413],
            ['H', 'H', 1, 0.74, 436],
            ['H', 'N', 1, 1.01, 391],
            ['H', 'O', 1, 0.96, 459],
            ['H', 'F', 1, 0.92, 565],
            # C-N bonds
            ['C', 'N', 1, 1.47, 308],
            ['C', 'N', 1, 1.39, 350],  #incorrect bond energy
            ['C', 'N', 1.5, 1.37, 450],  #pyrole
            ['C', 'N', 1.5, 1.33, 450],  #amide,pyridines
            # ['C', 'N', 2, 1.35, 450],
            ['C', 'N', 2, 1.27, 615],  #imines
            ['C', 'N', 3, 1.16, 887],
            ['C', 'F', 1, 1.35, 205],
            # C-O bonds
            ['C', 'O', 1, 1.47, 360],
            ['C', 'O', 1, 1.43, 360],
            ['C', 'O', 1, 1.36, 360],
            ['C', 'O', 1.5, 1.28, 650],
            ['C', 'O', 2, 1.23, 799],
            ['C', 'O', 2, 1.16, 1072],
            # C-C bonds
            ['C', 'C', 1, 1.54, 348],
            ['C', 'C', 1.5, 1.39, 518],
            ['C', 'C', 2, 1.34, 614],
            ['C', 'C', 3, 1.20, 839],
            # N-O
            ['N', 'O', 1, 1.4, 201],
            ['N', 'O', 1.5, 1.25, 451],
            ['N', 'O', 2, 1.21, 607],
            ['N', 'N', 1, 1.45, 167],
            ['N', 'N', 1.5, 1.32, 340],
            ['N', 'N', 2, 1.25, 418],
            ['N', 'N', 3, 1.10, 942],
            ['N', 'F', 1, 1.36, 283],
            # o-o
            ['O', 'O', 1, 1.48, 142],
            ['O', 'O', 2, 1.21, 494],
            ['O', 'F', 1, 1.42, 190],
            # F
            ['F', 'F', 1, 1.42, 155],
        ],
        columns=['atom_0', 'atom_1', 'bond_type', 'standard_bond_length', 'standard_bond_energy'],
    )

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
