from unittest.mock import patch
from id_features import IdFeatures
import pandas as pd
from .test_utils import get_dummy_items


def dummy(*args):
    pass


@patch('id_features.IdFeatures.fit', side_effect=dummy)
def test_fm_features_should_work_for_item_id(mock_fit):
    sales_df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 1, 1],
            [33, 1, 2],
            [34, 2, 1],
            [32, 2, 2],
            [30, 2, 2],
            [35, 3, 2],
            # now the validation set.
            [33, 2, 1],
            [36, 2, 1],
            [33, 1, 1],
        ],
        columns=['date_block_num', 'item_id', 'shop_id'])

    idf = IdFeatures(sales_df, get_dummy_items())
    assert mock_fit.is_called

    df = sales_df.copy()
    df.index += 10

    output_df = idf.get_fm_features(df)
    expected_data = [
        [0, True],
        [0, True],
        [1, False],
        [4, False],
        [2, False],
        [0, True],
        [0, True],
        [3, False],
        [6, False],
        [1, False],
    ]
    expected_df = pd.DataFrame(expected_data, index=df.index, columns=['item_id_oldness', 'item_id_is_fm'])
    assert output_df[expected_df.columns].equals(expected_df)
    assert output_df[df.columns].equals(df)


@patch('id_features.IdFeatures.fit', side_effect=dummy)
def test_fm_features_should_work_for_item_id_shop_id(mock_fit):
    item_id_and_shop_id = True
    sales_df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 1, 1],
            [33, 1, 2],
            [34, 2, 1],
            [32, 2, 2],
            [30, 2, 2],
            [35, 3, 2],
            # now the validation set.
            [33, 2, 1],
            [36, 2, 1],
            [33, 1, 1],
        ],
        columns=['date_block_num', 'item_id', 'shop_id'])

    idf = IdFeatures(sales_df, get_dummy_items())
    assert mock_fit.is_called

    df = sales_df.copy()
    df.index += 10

    output_df = idf.get_fm_features(df, item_id_and_shop_id=item_id_and_shop_id)
    expected_data = [
        [0, True],
        [0, True],
        [0, True],
        [1, False],
        [2, False],
        [0, True],
        [0, True],
        [0, True],
        [3, False],
        [1, False],
    ]
    expected_df = pd.DataFrame(
        expected_data, index=df.index, columns=['item_id_shop_id_oldness', 'item_id_shop_id_is_fm'])

    assert output_df[expected_df.columns].equals(expected_df)
    assert output_df[df.columns].equals(df)


@patch('id_features.IdFeatures.fit', side_effect=dummy)
def test_fm_features_should_work_for_unknowns(mock_fit):
    sales_df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 1, 1],
            [33, 1, 2],
            [34, 2, 1],
            [32, 2, 2],
            [30, 2, 2],
            [35, 3, 2],
            # now the validation set.
            [33, 2, 1],
            [36, 2, 1],
            [33, 1, 1],
        ],
        columns=['date_block_num', 'item_id', 'shop_id'])

    idf = IdFeatures(sales_df, get_dummy_items())
    assert mock_fit.is_called

    df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 4, 1],
            [33, 1, 2],
            [34, 2, 5],
        ], columns=['date_block_num', 'item_id', 'shop_id'])

    output_df = idf.get_fm_features(df)
    expected_data = [
        [0, True],
        [0, True],
        [1, False],
        [4, False],
    ]
    expected_df = pd.DataFrame(expected_data, index=df.index, columns=['item_id_oldness', 'item_id_is_fm'])
    assert output_df[expected_df.columns].equals(expected_df)
    assert output_df[df.columns].equals(df)
