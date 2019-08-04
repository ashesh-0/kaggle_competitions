from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# from decorators import timer


def get_obabel_atom_df(fname, structures_df):
    # OBABEL_DATA_DIR + 'openbabel.hdf', 'atoms'
    obabel_atom_features_df = pd.read_hdf(fname, 'atoms')
    obabel_atom_features_df['Type'] = LabelEncoder().fit_transform(obabel_atom_features_df['Type'])

    enc = LabelEncoder()
    enc.fit(structures_df.molecule_name)
    obabel_atom_features_df['molecule_name'] = enc.inverse_transform(obabel_atom_features_df.mol_id)
    cols_to_drop = ['X', 'Z', 'mol_id', 'Y']
    obabel_atom_features_df.drop(cols_to_drop, axis=1, inplace=True)
    float32_cols = obabel_atom_features_df.dtypes[obabel_atom_features_df.dtypes == np.float32].index.tolist()
    obabel_atom_features_df[float32_cols] = obabel_atom_features_df[float32_cols].astype(np.float16)
    return obabel_atom_features_df


@timer('ObabelBasedFeatures')
def add_obabel_based_features(X_df, obabel_atom_df, atom_index='atom_index_1'):
    """
    In train data, atom_index_0 is Hydrogen which does not have any useful information. Hence, atom_index is set to be
    atom_index_1
    """
    assert 'molecule_name' in X_df.columns
    X_df.reset_index(inplace=True)
    X_df = pd.merge(
        X_df,
        obabel_atom_df,
        how='left',
        left_on=['molecule_name', atom_index],
        right_on=['molecule_name', 'atom_index'])
    X_df.set_index('id', inplace=True)
    X_df.drop(['atom_index'], axis=1, inplace=True)
    return X_df


def compute_and_save():
    from open_babel_based_edges import write_edges_to_file
    from open_babel_based_atom_features import write_atom_features_to_file

    DATA_DIR = '/home/ashesh/Documents/initiatives/kaggle_competitions/predicting_molecular_properties/data/'

    enc = LabelEncoder()
    structures_df = pd.read_csv(DATA_DIR + 'structures.csv')
    enc.fit(structures_df.molecule_name)

    outputfname = DATA_DIR + 'openbabel.hdf'
    if os.path.exists(outputfname):
        os.remove(outputfname)

    write_edges_to_file(outputfname, 'edges', DATA_DIR + 'structures/', enc)
    print('Edge features written to file.')
    write_atom_features_to_file(outputfname, 'atoms', DATA_DIR + 'structures/', enc)
    print('Atom features written to file.')


# if __name__ == '__main__':
#     compute_and_save()
