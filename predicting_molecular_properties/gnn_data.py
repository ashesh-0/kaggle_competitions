import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from common_utils_molecule_properties import get_symmetric_edges
import numpy as np
from tqdm import tqdm_notebook


def _get_openbabel_based_atom_data(obabel_fname, structures_df):
    obabel_atoms_df = pd.read_hdf(obabel_fname, 'atoms')
    obabel_atoms_df = obabel_atoms_df[obabel_atoms_df['mol_id'].isin(structures_df['mol_id'].unique())]

    # Type encapsulates double bond, triple bond along with atom type.
    # Can be improved to decouple the two.
    type_list = obabel_atoms_df['Type'].unique().tolist()
    type_list.sort()
    obabel_atoms_df[[f'Type_{t}' for t in type_list]] = pd.get_dummies(obabel_atoms_df['Type'])
    obabel_atoms_df.drop('Type', axis=1, inplace=True)

    structures_df[['atom_C', 'atom_F', 'atom_H', 'atom_N', 'atom_O']] = pd.get_dummies(structures_df['atom'])
    atoms_df = pd.merge(structures_df, obabel_atoms_df, how='inner', on=['mol_id', 'atom_index'])
    atoms_df.sort_values('mol_id', inplace=True)

    atom_feature_cols = [
        'atom_C', 'atom_F', 'atom_H', 'atom_N', 'atom_O', 'SpinMultiplicity', 'Valence', 'HvyValence', 'PartialCharge',
        'MemberOfRingCount', 'MemberOfRingSize', 'CountRingBonds', 'SmallestBondAngle', 'AverageBondAngle',
        'IsHbondAcceptor', 'HasDoubleBond', 'Type_C+', 'Type_C1', 'Type_C2', 'Type_C3', 'Type_Cac', 'Type_Car',
        'Type_F', 'Type_HC', 'Type_HO', 'Type_N1', 'Type_N2', 'Type_N3', 'Type_N3+', 'Type_Nam', 'Type_Nar', 'Type_Ng+',
        'Type_Nox', 'Type_Npl', 'Type_Ntr', 'Type_O.co2', 'Type_O2', 'Type_O3'
    ]

    print('No of obabel based features', len(atom_feature_cols))
    return atoms_df[atom_feature_cols + ['mol_id', 'atom_index']]


def get_gnn_data(obabel_fname, structures_df, raw_X_df, atom_count=29, edge_scaler=None, atom_scaler=None):
    enc = LabelEncoder()
    structures_df['mol_id'] = enc.fit_transform(structures_df['molecule_name'])
    raw_X_df['mol_id'] = enc.transform(raw_X_df['molecule_name'])
    structures_df = structures_df[structures_df.mol_id.isin(raw_X_df.mol_id.unique())].copy()
    assert set(structures_df.mol_id.unique()) == set(raw_X_df.mol_id.unique())

    atom_data = get_atom_data(obabel_fname, structures_df, raw_X_df, atom_count=atom_count, scaler=atom_scaler)
    print('Atom data computed')
    edge_data = get_edge_data(obabel_fname, structures_df, raw_X_df, atom_count=atom_count, scaler=edge_scaler)
    print('Edge data computed')
    return {'atom': atom_data, 'edge': edge_data}


def get_atom_data(obabel_fname, structures_df, raw_X_df, atom_count=29, scaler=None):

    atoms_df = _get_openbabel_based_atom_data(obabel_fname, structures_df)
    mol_count = len(structures_df.mol_id.unique())

    mol_id_df = atoms_df.groupby('mol_id').size()
    start_indices = mol_id_df.cumsum().shift(1).fillna(0).astype(int).values

    atom_feature_cols = atoms_df.columns.tolist()
    atom_feature_cols.remove('mol_id')
    atom_feature_cols.remove('atom_index')

    # Normalize data.
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(atoms_df[atom_feature_cols])

    atoms_df[atom_feature_cols] = scaler.transform(atoms_df[atom_feature_cols])

    atom_data = atoms_df[atom_feature_cols].values

    atom_index_data = atoms_df[['atom_index']].values.astype(int)

    atom_features = np.zeros((mol_count, atom_count, len(atom_feature_cols)), dtype=np.float32)

    for i, start_index in enumerate(start_indices):
        end_index = start_indices[i + 1] if i + 1 < len(start_indices) else mol_count
        temp_idx = atom_index_data[start_index:end_index]
        atom_features[i, temp_idx[:, 0], :] = atom_data[start_index:end_index]

    return {'data': atom_features, 'scaler': scaler, 'mol_id': mol_id_df.index.tolist()}


def get_edge_data(obabel_fname, structures_df, raw_X_df, atom_count=29, scaler=None):
    # obabel_feature_cols = [
    #     'BondLength', 'EqubBondLength', 'BondOrder', 'IsAromatic', 'IsInRing', 'IsRotor', 'IsAmide', 'IsPrimaryAmide',
    #     'IsSecondaryAmide', 'IsTertiaryAmide', 'IsEster', 'IsCarbonyl', 'IsSingle', 'IsDouble', 'IsTriple', 'IsClosure',
    #     'IsUp', 'IsDown', 'IsWedge', 'IsHash', 'IsWedgeOrHash', 'IsCisOrTrans', 'IsDoubleBondGeometry'
    # ]
    obabel_feature_cols = [
        'BondLength', 'EqubBondLength', 'IsAromatic', 'IsInRing', 'IsCarbonyl', 'IsSingle', 'IsDouble', 'IsTriple',
        'IsClosure', 'IsCisOrTrans', 'IsDoubleBondGeometry'
    ]

    X_feature_cols = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
    raw_X_df[X_feature_cols] = pd.get_dummies(raw_X_df['type'])

    obabel_edges_df = pd.read_hdf(obabel_fname, 'edges')
    obabel_edges_df = obabel_edges_df[obabel_edges_df.mol_id.isin(raw_X_df.mol_id.unique())].copy()
    obabel_edges_df = get_symmetric_edges(obabel_edges_df)

    # Normalize data.
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(obabel_edges_df[obabel_feature_cols])

    obabel_edges_df[obabel_feature_cols] = scaler.transform(obabel_edges_df[obabel_feature_cols])

    mol_count = len(structures_df.mol_id.unique())

    raw_X_df = get_symmetric_edges(raw_X_df)

    obabel_edges_df.sort_values('mol_id', inplace=True)
    raw_X_df.sort_values('mol_id', inplace=True)

    X_start_id_df = raw_X_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).to_frame('X')
    obabel_start_id_df = obabel_edges_df.groupby('mol_id').size().cumsum().shift(1).fillna(0).to_frame('obabel')
    mol_start_idx = pd.concat([X_start_id_df, obabel_start_id_df], axis=1)[['X', 'obabel']].astype(int).values

    obabel_atom_index_data = obabel_edges_df[['atom_index_0', 'atom_index_1']].astype(np.int32).values
    X_atom_index_data = raw_X_df[['atom_index_0', 'atom_index_1']].astype(np.int32).values

    obabel_data = obabel_edges_df[obabel_feature_cols].values
    X_data = raw_X_df[X_feature_cols].values
    target_data = raw_X_df['scalar_coupling_constant'].values

    target_edge_features = np.zeros((mol_count, atom_count, atom_count, 1), dtype=np.float32)

    edge_features = np.zeros(
        (mol_count, atom_count, atom_count, len(X_feature_cols) + len(obabel_feature_cols)), dtype=np.float32)

    for i, both_ids in enumerate(tqdm_notebook(mol_start_idx)):
        X_start_index = both_ids[0]
        X_end_index = mol_start_idx[i + 1, 0] if i + 1 < len(mol_start_idx) else raw_X_df.shape[0]
        temp_X_ids = X_atom_index_data[X_start_index:X_end_index]

        obabel_start_index = both_ids[1]
        obabel_end_index = mol_start_idx[i + 1, 1] if i + 1 < len(mol_start_idx) else obabel_edges_df.shape[0]
        temp_obabel_ids = obabel_atom_index_data[obabel_start_index:obabel_end_index]

        # Target feature.
        target_edge_features[i, temp_X_ids[:, 0], temp_X_ids[:, 1], 0] = target_data[X_start_index:X_end_index]

        edge_features[i, temp_X_ids[:, 0], temp_X_ids[:, 1], :len(X_feature_cols)] = X_data[X_start_index:X_end_index]
        edge_features[i, temp_obabel_ids[:, 0], temp_obabel_ids[:, 1],
                      len(X_feature_cols):] = obabel_data[obabel_start_index:obabel_end_index]

    return {
        'data': edge_features,
        'scaler': scaler,
        'target': target_edge_features,
        'mol_id': obabel_start_id_df.index.tolist(),
    }
