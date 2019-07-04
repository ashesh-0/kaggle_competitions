# from intermediate_atoms import add_intermediate_atom_stats
from distance_features import add_distance_features
from molecule_features import add_molecule_features
from neighbor_features_atom_index import add_neighbors_features
from bond_features import add_bond_features
import pandas as pd


def get_X(X_df, structures_df, atom_encoder):
    X_df = add_molecule_features(X_df, structures_df)
    # It is necessary to first call molecule feature as distance features use some of the columns created in
    # molecule features.
    X_df = add_distance_features(X_df, structures_df)
    # it must be called after distance features
    X_df = add_bond_features(X_df)
    # it must be called after distance features.
    X_df = add_neighbors_features(X_df, structures_df, atom_encoder)
    # X_df = add_intermediate_atom_stats(X_df, structures_df)
    bond_encoding(X_df)
    return X_df


def data_augmentation_on_train(train_X_df, train_Y_df):
    """
    Since target is agnostic of the order of the atoms, (H-O and O-H will have same constant)
    """
    X_aug = train_X_df[train_X_df['enc_atom_0'] != train_X_df['enc_atom_1']].copy()
    Y_aug = train_Y_df.loc[X_aug.index].copy()

    for col in ['atom_index', 'x', 'y', 'z', 'enc_atom']:
        zero_col = col + '_0'
        one_col = col + '_1'
        X_aug[zero_col] = train_X_df[one_col]
        X_aug[one_col] = train_X_df[zero_col]

    X_aug.index = X_aug.index + max(train_X_df.index) + 1
    Y_aug.index = Y_aug.index + max(train_Y_df.index) + 1

    X_final = pd.concat([train_X_df, X_aug])
    Y_final = pd.concat([train_Y_df, Y_aug])
    return (X_final, Y_final)


def bond_encoding(df):
    """
    ONLY 3 TYPE OF BONDS EXIST!! Not a very strong feature.
    Gives an id for the interation pair
    """
    bond_map = {
        'CC': 0,
        'CH': 1,
        'CN': 2,
        'CO': 3,
        'CF': 4,
        'HC': 1,
        'HH': 5,
        'HN': 6,
        'HO': 7,
        'HF': 8,
        'NC': 2,
        'NH': 6,
        'NN': 9,
        'NO': 10,
        'NF': 11,
        'OC': 3,
        'OH': 7,
        'ON': 10,
        'OO': 12,
        'OF': 13,
        'FC': 4,
        'FH': 8,
        'FN': 11,
        'FO': 13,
        'FF': 14
    }
    bond = df['atom_0'] + df['atom_1']
    df['enc_bond'] = bond.map(bond_map)


# if __name__ == '__main__':
#     DIR = '/home/ashesh/Documents/initiatives/kaggle_competitions/predicting_molecular_properties/data/'
#     X_df = pd.read_csv(DIR + 'train.csv', index_col=0)
#     structures_df = pd.read_csv(DIR + 'structures.csv')
#     X_df = add_distance_features(X_df, structures_df)
#     X_df = add_intermediate_atom_stats(X_df, structures_df)
