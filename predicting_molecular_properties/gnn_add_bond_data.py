from common_data_molecule_properties import get_bond_data
from common_utils_molecule_properties import add_structure_data_to_edge


def add_bond_data_to_edges_df(edges_df, structures_df):
    bond_df = get_bond_data()
    bond_df['atom_bond_type'] = bond_df['atom_0'] + bond_df['atom_1'] + bond_df['bond_type'].astype(str)
    bond_df = bond_df.groupby('atom_bond_type')[['standard_bond_energy', 'Electronegativity_diff']].mean()

    edges_df = add_structure_data_to_edge(edges_df, structures_df, ['atom'])
    edges_df['atom_bond_type'] = edges_df['atom_0'] + edges_df['atom_1'] + edges_df['bond_type'].astype(str)

    output_df = edges_df.join(bond_df, how='left', on='atom_bond_type')
    edges_df.drop(['atom_0', 'atom_1', 'atom_bond_type'], axis=1, inplace=True)
    return output_df
