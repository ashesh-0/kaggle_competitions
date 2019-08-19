import pandas as pd
from numpy_angle_computation import angle_based_stats


def test_compute_angle_one_molecule():
    structures_df = pd.DataFrame(
        [
            [0, 1, 0, 0, 0],
            [0, 2, 0, 1, 0],
            [0, 3, 1, 1, 0],
            [0, 4, 1, 0, 0],
            [0, 5, 0.5, 0.5, 1],
        ],
        columns=['mol_id', 'atom_index', 'x', 'y', 'z'])

    nbr_df = pd.DataFrame(
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
        ], columns=['mol_id', 'atom_index', 'nbr_atom_index'])

    df = angle_based_stats(structures_df, nbr_df)
    print(df)
