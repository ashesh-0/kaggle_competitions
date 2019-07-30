from common_data_molecule_properties import get_electonegativity
from common_utils_molecule_properties import add_structure_data_to_edge
import pandas as pd

# def _add_data_to_edges(structures_df, edges_df):
#     lone_pair = pd.DataFrame.from_dict({'C': 0, 'H': 0, 'F': 1, 'N': 1, 'O': 1}, orient='index', columns=['lone_pair'])
#     lone_pair.index.name = 'atom'
#     elec_df = get_electonegativity()
#     data_df = pd.concat([lone_pair, elec_df], axis=1, sort=True)

#     edges_df = add_structure_data_to_edge(edges_df, structures_df)

#     data_df.index.name = 'atom_0'
#     edges_df = edges_df.join(data_df, how='left', on='atom_0')
#     edges_df.rename({'lone_pair': 'lone_pair_0', 'Electronegativity': 'Electronegativity_0'}, axis=1, inplace=True)

#     data_df.index.name = 'atom_1'
#     edges_df = edges_df.join(data_df, how='left', on='atom_1')
#     edges_df.rename({'lone_pair': 'lone_pair_1', 'Electronegativity': 'Electronegativity_1'}, axis=1, inplace=True)

#     edges_df['Electronegativity_diff_0-1'] = edges_df['Electronegativity_0'] - edges_df['Electronegativity_1']
#     return edges_df


def _get_pi_donor_feature(structures_df, edges_df):
    """
    Returns for each molecule,atom_index what is the pi donor property of that atom. This is not a  feature for this
    atom_index. Rather, it is a feature of all its 2 neighbors.
    """
    # Remove H
    edges_df['Electronegativity_diff_0-1'] = edges_df['Electronegativity_0'] - edges_df['Electronegativity_1']
    edges_df = edges_df[(edges_df['atom_0'] != 'H') & (edges_df['atom_1'] != 'H')]
    df = edges_df.groupby(['molecule_name', 'atom_index_0']).agg({
        'atom_0_lone_pair': 'first',
        'Electronegativity_diff_0-1': 'sum'
    })

    df['pi_donor_0'] = df['atom_0_lone_pair'] + df['Electronegativity_diff_0-1']
    df['pi_donor_1'] = df['atom_0_lone_pair'] * 0.5 + df['Electronegativity_diff_0-1']
    df['pi_donor_2'] = df['atom_0_lone_pair'] * 2 + df['Electronegativity_diff_0-1']
    df.reset_index(inplace=True)
    df.rename({'atom_index_0': 'atom_index'}, axis=1, inplace=True)

    return df[['molecule_name', 'atom_index', 'pi_donor_0', 'pi_donor_1', 'pi_donor_2']]


def get_pi_donor_feature(structures_df, edges_df, neighbors_df):
    """
    Returns the feature value for each molecule_name,atom_index. this can now be merged with X_df on atom_index_[0\1]
    to get meaningful feature.
    """
    assert 'atom_index_0' in neighbors_df.columns
    assert 'atom_index_1' in neighbors_df.columns
    assert 'molecule_name' in neighbors_df.columns

    feature_df = _get_pi_donor_feature(structures_df, edges_df)
    neighbors_df = neighbors_df[neighbors_df.nbr_distance == 2]
    neighbors_df = pd.merge(
        neighbors_df,
        feature_df,
        how='inner',
        left_on=['molecule_name', 'atom_index_1'],
        right_on=['molecule_name', 'atom_index'],
    )
    feature_df = neighbors_df.groupby(['molecule_name', 'atom_index_0'])[['pi_donor_0', 'pi_donor_1',
                                                                          'pi_donor_2']].sum()
    feature_df.reset_index(inplace=True)
    return feature_df
