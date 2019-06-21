import numpy as np
import pandas as pd
from intermediate_atoms import _find_distance_from_plane, _add_plane_vector, add_intermediate_atom_stats
from distance_features import add_distance_features


def test_find_distance_from_plane():
    df = pd.DataFrame([[1, 1, 1, -3, np.sqrt(3)]], columns=['m_x', 'm_y', 'm_z', 'c', 'm_2norm'])
    df['x'] = 0
    df['y'] = 0
    df['z'] = 0
    d = _find_distance_from_plane(df, 'x', 'y', 'z')
    assert np.abs(np.abs(d.iloc[0]) - np.sqrt(3)) < 1e-10
    assert d.shape[0] == 1


def test_add_plane_vector():
    df = pd.DataFrame(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        columns=['x_1', 'y_1', 'z_1', 'x_0', 'y_0', 'z_0'])

    _add_plane_vector(df)
    assert df.iloc[0]['m_x'] == -1
    assert df.iloc[0]['m_y'] == 0
    assert df.iloc[0]['m_z'] == 0
    assert df.iloc[0]['c'] == 1

    assert df.iloc[1]['m_x'] == 0
    assert df.iloc[1]['m_y'] == -1
    assert df.iloc[1]['m_z'] == 0
    assert df.iloc[1]['c'] == 1

    assert df.iloc[2]['m_x'] == -1
    assert df.iloc[2]['m_y'] == -1
    assert df.iloc[2]['m_z'] == -1
    assert df.iloc[2]['c'] == 3


def _dummy_structures():
    data = [
        ['dsgdb9nsd_000001', 0, 'C', -0.012698135900000001, 1.085804158, 0.008000995799999999],
        ['dsgdb9nsd_000001', 1, 'H', 0.002150416, -0.0060313176, 0.0019761204],
        ['dsgdb9nsd_000001', 2, 'H', 1.011730843, 1.4637511619999999, 0.0002765748],
        ['dsgdb9nsd_000001', 3, 'H', -0.540815069, 1.447526614, -0.8766437152],
        ['dsgdb9nsd_000001', 4, 'H', -0.5238136345000001, 1.437932644, 0.9063972942],
        # 2nd molecule
        ['dsgdb9nsd_000003', 0, 'O', -0.0343604951, 0.9775395708, 0.0076015922999999996],
        ['dsgdb9nsd_000003', 1, 'H', 0.0647664923, 0.020572198899999998, 0.0015346341],
        ['dsgdb9nsd_000003', 2, 'H', 0.8717903737, 1.300792405, 0.0006931336],
    ]
    structures_df = pd.DataFrame(data, columns=['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z'])
    train_df = pd.DataFrame(
        [
            ['dsgdb9nsd_000001', 3, 4],
            ['dsgdb9nsd_000001', 4, 0],
            ['dsgdb9nsd_000003', 1, 2],
        ],
        columns=['molecule_name', 'atom_index_0', 'atom_index_1'])
    train_df = add_distance_features(train_df, structures_df)
    return (train_df, structures_df)


def test_add_intermediate_atom_stats():
    train_df, structures_df = _dummy_structures()
    print('')
    print(structures_df.T)
    output_df = add_intermediate_atom_stats(train_df.copy(), structures_df, x1_distance_fraction=1)
    assert output_df.index.equals(train_df.index)

    assert output_df.iloc[0]['intm_O_plane_cnt'] == 0
    assert output_df.iloc[0]['intm_C_plane_cnt'] == 1
    assert output_df.iloc[0]['intm_H_plane_cnt'] == 2
    assert output_df.iloc[0]['intm_total_plane_cnt'] == 3

    assert output_df.iloc[1]['intm_O_plane_cnt'] == 0
    assert output_df.iloc[1]['intm_C_plane_cnt'] == 0
    assert output_df.iloc[1]['intm_H_plane_cnt'] == 0
    assert output_df.iloc[1]['intm_total_plane_cnt'] == 0

    assert output_df.iloc[2]['intm_O_plane_cnt'] == 1
    assert output_df.iloc[2]['intm_C_plane_cnt'] == 0
    assert output_df.iloc[2]['intm_H_plane_cnt'] == 0
    assert output_df.iloc[2]['intm_total_plane_cnt'] == 1
    print(output_df.T)
