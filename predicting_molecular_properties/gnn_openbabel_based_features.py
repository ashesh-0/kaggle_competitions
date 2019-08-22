import pandas as pd


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
        'atom_C', 'atom_O', 'SpinMultiplicity', 'Valence', 'HvyValence', 'PartialCharge', 'MemberOfRingCount',
        'MemberOfRingSize', 'SmallestBondAngle', 'AverageBondAngle', 'IsHbondAcceptor', 'Type_C2', 'Type_C3',
        'Type_Car', 'Type_HC', 'Type_O2', 'Type_O3'
    ]

    # atom_feature_cols = [
    #     'atom_C', 'atom_F', 'atom_H', 'atom_N', 'atom_O', 'SpinMultiplicity', 'Valence', 'HvyValence', 'PartialCharge',
    #     'MemberOfRingCount', 'MemberOfRingSize', 'CountRingBonds', 'SmallestBondAngle', 'AverageBondAngle',
    #     'IsHbondAcceptor', 'HasDoubleBond', 'Type_C+', 'Type_C1', 'Type_C2', 'Type_C3', 'Type_Cac', 'Type_Car',
    #     'Type_F', 'Type_HC', 'Type_HO', 'Type_N1', 'Type_N2', 'Type_N3', 'Type_N3+', 'Type_Nam', 'Type_Nar', 'Type_Ng+',
    #     'Type_Nox', 'Type_Npl', 'Type_Ntr', 'Type_O.co2', 'Type_O2', 'Type_O3'
    # ]

    print('No of obabel based features', len(atom_feature_cols))
    return atoms_df[atom_feature_cols + ['mol_id', 'atom_index']]
