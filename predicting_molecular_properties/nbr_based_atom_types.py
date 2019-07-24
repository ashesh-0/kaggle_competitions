"""
Every atom of a molecule will be given a type depending upon the type of bonds it is making.
Different types of Oxygen:
    C=0
    C-O-H
    C-O-C
    C-O-N
    H-O-N
    N=O
"""
import pandas as pd
from common_utils_molecule_properties import add_structure_data_to_edge
from sklearn.preprocessing import LabelEncoder
import numpy as np

# from decorators import timer


def _get_feature(count_df, atom_valency):
    total_dict = {
        1: count_df['total'] == 1,
        2: count_df['total'] == 2,
        3: count_df['total'] == 3,
        4: count_df['total'] == 4,
    }
    C_dict = {
        0: count_df['C'] == 0,
        1: count_df['C'] == 1,
        2: count_df['C'] == 2,
        3: count_df['C'] == 3,
        4: count_df['C'] == 4,
    }
    N_dict = {
        0: count_df['N'] == 0,
        1: count_df['N'] == 1,
        2: count_df['N'] == 2,
        3: count_df['N'] == 3,
        4: count_df['N'] == 4,
    }
    O_dict = {
        0: count_df['O'] == 0,
        1: count_df['O'] == 1,
        2: count_df['O'] == 2,
        3: count_df['O'] == 3,
        4: count_df['O'] == 4,
    }
    H_dict = {
        0: count_df['H'] == 0,
        1: count_df['H'] == 1,
        2: count_df['H'] == 2,
        3: count_df['H'] == 3,
        4: count_df['H'] == 4,
    }

    F_dict = {
        0: count_df['F'] == 0,
        1: count_df['F'] == 1,
        2: count_df['F'] == 2,
        3: count_df['F'] == 3,
        4: count_df['F'] == 4,
    }

    df = pd.DataFrame([], index=count_df.index)
    # C,F,H,N,O
    for c in range(atom_valency + 1):
        filtr_C = True if c == 0 else C_dict[c]
        for f in range(atom_valency + 1):
            if c + f > atom_valency:
                continue
            filtr_CF = filtr_C if f == 0 else filtr_C & F_dict[f]
            for h in range(atom_valency + 1):
                if c + f + h > atom_valency:
                    continue
                filtr_CFH = filtr_CF if h == 0 else filtr_CF & H_dict[h]

                for n in range(atom_valency + 1):
                    if c + f + h + n > atom_valency:
                        continue
                    filtr_CFHN = filtr_CFH if n == 0 else filtr_CFH & N_dict[n]

                    for o in range(atom_valency + 1):
                        tot = c + f + h + n + o
                        if tot > atom_valency or tot == 0:
                            continue
                        filtr_CFHNO = filtr_CFHN if o == 0 else filtr_CFHN & O_dict[o]
                        filtr = filtr_CFHNO & total_dict[tot]
                        name = f'C{c}F{f}H{h}N{n}O{o}'
                        df[name] = filtr

    return df


def get_sorted_types():
    """
    sorted by number of occurances.
    """
    dtypes = [
        'C0F0H0N0O1', 'C0F0H0N0O2', 'C0F0H0N0O3', 'C0F0H0N1O0', 'C0F0H0N1O1', 'C0F0H0N1O2', 'C0F0H0N2O0', 'C0F0H0N2O1',
        'C0F0H0N3O0', 'C0F0H1N0O0', 'C0F0H1N0O2', 'C0F0H1N1O0', 'C0F0H1N1O1', 'C0F0H1N2O0', 'C0F0H2N0O0', 'C0F0H2N0O2',
        'C0F0H2N1O0', 'C0F0H3N0O0', 'C0F0H3N0O1', 'C0F0H3N1O0', 'C0F0H4N0O0', 'C0F1H0N1O1', 'C0F1H0N2O0', 'C1F0H0N0O0',
        'C1F0H0N0O1', 'C1F0H0N0O2', 'C1F0H0N1O0', 'C1F0H0N1O1', 'C1F0H0N2O0', 'C1F0H1N0O0', 'C1F0H1N0O1', 'C1F0H1N0O2',
        'C1F0H1N1O0', 'C1F0H2N0O0', 'C1F0H2N0O1', 'C1F0H2N1O0', 'C1F0H3N0O0', 'C1F1H0N0O1', 'C1F1H0N1O0', 'C1F3H0N0O0',
        'C2F0H0N0O0', 'C2F0H0N0O1', 'C2F0H0N0O2', 'C2F0H0N1O0', 'C2F0H1N0O0', 'C2F0H1N0O1', 'C2F0H1N1O0', 'C2F0H2N0O0',
        'C2F1H0N0O0', 'C3F0H0N0O0', 'C3F0H0N0O1', 'C3F0H0N1O0', 'C3F0H1N0O0', 'C4F0H0N0O0'
    ]
    return dtypes


# @timer('NbrBasedAtomTypes')
def add_atom_type_both_indices(df, atom_type_df):
    df.reset_index(inplace=True)

    df = add_atom_type(df, atom_type_df, 'atom_index_0')
    df.rename({'neighbor_type': 'atom_index_0_neighbor_type'}, axis=1, inplace=True)

    df = add_atom_type(df, atom_type_df, 'atom_index_1')
    df.rename({'neighbor_type': 'atom_index_1_neighbor_type'}, axis=1, inplace=True)
    df.set_index('id', inplace=True)
    return df


def add_atom_type(df, atom_type_df, atom_index_col):
    """
    'neighbor_type' column gets added.
    """
    atom_type_df = atom_type_df.rename({'atom_index': 'atom_index__'}, axis=1)
    df = pd.merge(
        df,
        atom_type_df,
        how='left',
        left_on=['molecule_name', atom_index_col],
        right_on=['molecule_name', 'atom_index__'])
    df.drop('atom_index__', axis=1, inplace=True)
    return df


def get_atom_type(edges_df, structures_df):
    edges_df = add_structure_data_to_edge(edges_df, structures_df)
    # It has atom_0,atom_1
    edges_df['H'] = edges_df['atom_1'] == 'H'
    edges_df['C'] = edges_df['atom_1'] == 'C'
    edges_df['O'] = edges_df['atom_1'] == 'O'
    edges_df['N'] = edges_df['atom_1'] == 'N'
    edges_df['F'] = edges_df['atom_1'] == 'F'

    edges_df.rename(
        {
            'atom_index_0': 'atom_index',
            'atom_0': 'atom',
        }, axis=1, inplace=True)

    df = edges_df.groupby(['molecule_name', 'atom_index']).agg({
        'H': 'sum',
        'C': 'sum',
        'O': 'sum',
        'N': 'sum',
        'F': 'sum',
        'atom': 'first',
    })
    df['total'] = df[['H', 'C', 'O', 'N', 'F']].sum(axis=1)

    # Carbon atom types
    df_C = df[df['atom'] == 'C']
    feature_C = _get_feature(df_C, 4)
    print('Carbon atom neighbor atom types counted')

    # Oxygen atom types.
    df_O = df[df['atom'] == 'O']
    feature_O = _get_feature(df_O, 2)
    print('Oxygen atom neighbor atom types counted')
    # Nitrogen atom types
    df_N = df[df['atom'] == 'N']
    feature_N = _get_feature(df_N, 3)
    print('Nitrogen atom neighbor atom types counted')

    # Hydrogen
    df_H = df[df['atom'] == 'H']
    feature_H = _get_feature(df_H, 1)
    print('Hydrogen atom neighbor atom types counted')

    # Florine
    df_F = df[df['atom'] == 'F']
    feature_F = _get_feature(df_F, 1)
    print('Florine atom neighbor atom types counted')

    feature_df = pd.concat([feature_C, feature_N, feature_O, feature_H, feature_F], axis=0, sort=True).fillna(False)
    feature_df.columns.name = 'neighbor_type'
    feature_df = feature_df[feature_df == True]
    feature_df = feature_df.stack().reset_index()

    assert feature_df.shape[0] == feature_df.groupby(['molecule_name', 'atom_index']).first().shape[0]

    feature_df.drop(0, axis=1, inplace=True)
    feature_df.sort_values(['molecule_name', 'atom_index'], inplace=True)

    # from string to number
    types = set(feature_df['neighbor_type'].unique())
    assert types.issubset(set(get_sorted_types()))
    lb = LabelEncoder()
    lb.fit(get_sorted_types())
    feature_df['neighbor_type'] = lb.transform(feature_df['neighbor_type']).astype(np.int16)
    return feature_df
