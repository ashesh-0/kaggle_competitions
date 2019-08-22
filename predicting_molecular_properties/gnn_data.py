import pandas as pd
from common_utils import Scaler
from common_utils_molecule_properties import get_symmetric_edges
# from numpy_angle_computation import get_angle_based_features
# from gnn_edge_feature_computation import get_intermediate_angle_features
# from gnn_openbabel_based_features import _get_openbabel_based_atom_data
# from gnn_distance_based_features import induced_electronegativity_atom_feature, induced_electronegativity_feature
# from gnn_add_bond_data import add_bond_data_to_edges_df

import numpy as np
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder


def get_gnn_data(obabel_fname,
                 structures_df,
                 raw_X_df,
                 nbr_df,
                 edges_df,
                 ia_df,
                 conical_features_df,
                 atom_count=29,
                 edge_scaler=None,
                 atom_scaler=None):
    assert 'mol_id' in structures_df
    assert 'mol_id' in raw_X_df

    structures_df = structures_df[structures_df.mol_id.isin(raw_X_df.mol_id.unique())].copy()
    edges_df = edges_df[edges_df.mol_id.isin(raw_X_df.mol_id.unique())].copy()
    structures_df.sort_values(['mol_id', 'atom_index'], inplace=True)
    edges_df.sort_values(['mol_id'], inplace=True)

    assert set(structures_df.mol_id.unique()) == set(raw_X_df.mol_id.unique())

    edge_data = get_edge_data(
        obabel_fname,
        structures_df,
        raw_X_df,
        edges_df,
        ia_df,
        conical_features_df,
        atom_count=atom_count,
        scaler=edge_scaler)
    print('Edge data computed')
    atom_data = get_atom_data(
        obabel_fname, structures_df, raw_X_df, nbr_df, edges_df, atom_count=atom_count, scaler=atom_scaler)
    print('Atom data computed')

    return {'atom': atom_data, 'edge': edge_data}


def get_atom_data(obabel_fname, structures_df, raw_X_df, nbr_df, edges_df, atom_count=29, scaler=None, imputer=None):

    atoms_df = _get_openbabel_based_atom_data(obabel_fname, structures_df)
    angle_features_df = get_angle_based_features(structures_df, nbr_df, edges_df).reset_index()
    en_atom_df, _ = induced_electronegativity_atom_feature(structures_df, edges_df)

    atoms_df = pd.merge(atoms_df, angle_features_df, on=['mol_id', 'atom_index'], how='outer')
    atoms_df = pd.merge(atoms_df, en_atom_df, on=['mol_id', 'atom_index'], how='outer')

    # atoms_df = pd.concat([atoms_df, angle_features_df], axis=1)

    mol_count = len(structures_df.mol_id.unique())

    mol_id_df = atoms_df.groupby('mol_id').size()
    start_indices = mol_id_df.cumsum().shift(1).fillna(0).astype(int).values

    atom_feature_cols = atoms_df.columns.tolist()
    atom_feature_cols.remove('mol_id')
    atom_feature_cols.remove('atom_index')

    # Skip these from normalization.
    binary_features = [
        'atom_C', 'atom_O', 'IsHbondAcceptor', 'Type_C2', 'Type_C3', 'Type_Car', 'Type_HC', 'Type_O2', 'Type_O3'
    ]

    if scaler is None:
        scaler = Scaler(skip_columns=binary_features + ['mol_id', 'atom_index'], dtype=np.float16)
        scaler.fit(atoms_df)

    scaler.transform(atoms_df)

    atom_data = atoms_df[atom_feature_cols].values
    print('AtomFeatures:', len(atom_feature_cols), atom_feature_cols)

    atom_index_data = atoms_df[['atom_index']].values.astype(int)

    atom_features = np.zeros((mol_count, atom_count, len(atom_feature_cols)), dtype=np.float16)

    for i, start_index in enumerate(start_indices):
        end_index = start_indices[i + 1] if i + 1 < len(start_indices) else mol_count
        temp_idx = atom_index_data[start_index:end_index]
        atom_features[i, temp_idx[:, 0], :] = atom_data[start_index:end_index]

    return {'data': atom_features, 'imputer': imputer, 'scaler': scaler, 'mol_id': mol_id_df.index.tolist()}


def _get_edge_df(obabel_fname, structures_df, raw_X_df, raw_edges_df, ia_df, conical_features_df):

    # electronegativity and bond energy feature
    bond_data_df = add_bond_data_to_edges_df(raw_edges_df, structures_df)[[
        'mol_id', 'atom_index_0', 'atom_index_1', 'standard_bond_energy', 'Electronegativity_diff'
    ]]

    X_feature_cols = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']
    raw_X_df[X_feature_cols] = pd.get_dummies(raw_X_df['type'])
    raw_X_df['type'] = LabelEncoder().fit_transform(raw_X_df['type'])

    # add electronegativity features to edges data.
    en_data_df = induced_electronegativity_feature(structures_df, raw_edges_df, raw_X_df)['edge']
    raw_X_df = pd.concat([raw_X_df, en_data_df], axis=1)
    X_feature_cols += en_data_df.columns.tolist()

    obabel_feature_cols = ['BondLength', 'EqubBondLength', 'IsAromatic', 'IsInRing', 'IsSingle', 'IsDouble', 'IsTriple']

    obabel_edges_df = pd.read_hdf(obabel_fname,
                                  'edges')[['mol_id', 'atom_index_0', 'atom_index_1'] + obabel_feature_cols]

    obabel_edges_df = obabel_edges_df[obabel_edges_df.mol_id.isin(raw_X_df.mol_id.unique())].copy()
    dihedral_df = get_intermediate_angle_features(get_symmetric_edges(raw_edges_df), raw_X_df, structures_df, ia_df)

    raw_X_df = pd.concat([raw_X_df, dihedral_df], axis=1)
    raw_X_df.drop(['molecule_name'], axis=1, inplace=True)

    bond_data_df = get_symmetric_edges(bond_data_df)
    raw_X_df = get_symmetric_edges(raw_X_df)
    obabel_edges_df = get_symmetric_edges(obabel_edges_df)
    conical_features_df = get_symmetric_edges(conical_features_df)

    # Merging adds nans to some columns. bool turn to object, float16 turns to float64 and so on.
    edges_df = pd.merge(raw_X_df, obabel_edges_df, how='outer', on=['mol_id', 'atom_index_0', 'atom_index_1'])
    edges_df = pd.merge(edges_df, bond_data_df, how='outer', on=['mol_id', 'atom_index_0', 'atom_index_1'])

    edges_df = pd.merge(edges_df, conical_features_df, how='outer', on=['mol_id', 'atom_index_0', 'atom_index_1'])
    return edges_df


def get_edge_data(obabel_fname,
                  structures_df,
                  raw_X_df,
                  raw_edges_df,
                  ia_df,
                  conical_features_df,
                  atom_count=29,
                  scaler=None,
                  imputer=None):
    # obabel_feature_cols = [
    #     'BondLength', 'EqubBondLength', 'BondOrder', 'IsAromatic', 'IsInRing', 'IsRotor', 'IsAmide', 'IsPrimaryAmide',
    #     'IsSecondaryAmide', 'IsTertiaryAmide', 'IsEster', 'IsCarbonyl', 'IsSingle', 'IsDouble', 'IsTriple', 'IsClosure',
    #     'IsUp', 'IsDown', 'IsWedge', 'IsHash', 'IsWedgeOrHash', 'IsCisOrTrans', 'IsDoubleBondGeometry'
    # ]

    edges_df = _get_edge_df(obabel_fname, structures_df, raw_X_df, raw_edges_df, ia_df, conical_features_df)
    feature_cols = edges_df.columns.tolist()

    feature_cols.remove('mol_id')
    feature_cols.remove('atom_index_0')
    feature_cols.remove('atom_index_1')
    feature_cols.remove('scalar_coupling_constant')
    feature_cols.remove('type')

    print('Nan fraction')
    print(edges_df[feature_cols].isna().sum() / edges_df.shape[0])
    print('')
    print('EdgeFeatures:', len(feature_cols), feature_cols)

    # Normalize data.
    binary_features = [
        '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN', 'IsAromatic', 'IsInRing', 'IsSingle',
        'IsDouble', 'IsTriple', 'IsCisOrTrans'
    ]

    edges_df[binary_features] = edges_df[binary_features].fillna(0).astype(np.float16)

    if scaler is None:
        scaler = Scaler(
            skip_columns=binary_features +
            ['mol_id', 'atom_index_0', 'atom_index_1', 'scalar_coupling_constant', 'type'],
            dtype=np.float16,
        )
        scaler.fit(edges_df)

    scaler.transform(edges_df)

    mol_count = len(structures_df.mol_id.unique())

    edges_df.sort_values('mol_id', inplace=True)

    start_id_df = edges_df.groupby('mol_id').size().cumsum().shift(1).fillna(0)
    start_ids = start_id_df.astype(int).values
    atom_index_data = edges_df[['atom_index_0', 'atom_index_1']].astype(np.int32).values

    edge_data = edges_df[feature_cols].to_numpy(dtype=np.float16)
    target_data = edges_df[['scalar_coupling_constant', 'type']]
    target_data['scalar_coupling_constant'] = target_data['scalar_coupling_constant'].fillna(0)
    target_data['type'] = target_data['type'].fillna(-1)
    target_data = target_data.values

    target_edge_features = np.zeros((mol_count, atom_count, atom_count, 2), dtype=np.float16)
    # By default, type is -1
    target_edge_features[:, :, :, 1] = -1

    edge_features = np.zeros((mol_count, atom_count, atom_count, len(feature_cols)), dtype=np.float16)

    for i, start_index in enumerate(tqdm_notebook(start_ids)):
        end_index = start_ids[i + 1] if i + 1 < len(start_ids) else edges_df.shape[0]
        temp_ai_ids = atom_index_data[start_index:end_index]

        # Target feature.
        target_edge_features[i, temp_ai_ids[:, 0], temp_ai_ids[:, 1], :] = target_data[start_index:end_index, :]

        edge_features[i, temp_ai_ids[:, 0], temp_ai_ids[:, 1], :] = edge_data[start_index:end_index]

    return {
        'data': edge_features,
        'scaler': scaler,
        'target': target_edge_features,
        'mol_id': start_id_df.index.tolist(),
    }


# [['CONIC_REGION_-3_LonePairAtom', 'CONIC_REGION_-1_CountAtom', 'CONIC_REGION_-3_ValenceElectronsAtom', 'CONIC_REGION_-2_CountAtom', 'CONIC_REGION_-2_LonePairAtom'],
# ['CONIC_REGION_0_CountAtom', 'CONIC_REGION_1_MassAtom'],
# ['CONIC_REGION_0_Q', 'CONIC_REGION_-1_EN', 'CONIC_REGION_1_ValenceElectronsAtom', 'CONIC_REGION_-1_CountAtom'],
# ['CONIC_REGION_1_Q', 'CONIC_REGION_-3_LonePairAtom', 'CONIC_REGION_-2_LonePairAtom'],
# ['CONIC_REGION_-2_CountAtom', 'CONIC_REGION_1_CountAtom', 'CONIC_REGION_0_EN', 'CONIC_REGION_-2_EN', 'CONIC_REGION_1_Q', 'CONIC_REGION_0_CountAtom'],
# ['CONIC_REGION_-2_MassAtom', 'CONIC_REGION_2_ValenceElectronsAtom'],]

# top_k = 2
# output = []
# for c in data:
#     output += c[-top_k:]
# output = list(set(output))
# output.sort()
# output
