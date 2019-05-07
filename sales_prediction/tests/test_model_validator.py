import itertools
import pandas as pd
import numpy as np
from unittest.mock import patch
from model_validator import ModelValidator

ITEMS = [1, 2, 3]
SHOPS = [1, 2]


def mock_get_items(*args):
    return ITEMS


def mock_get_shops(*args):
    return SHOPS


@patch('model_validator.get_shops_in_market', side_effect=mock_get_shops)
@patch('model_validator.get_items_in_market', side_effect=mock_get_items)
def test_model_validator_should_make_last_month_as_validation(m1, m2):
    sales_df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 1, 1],
            [32, 1, 2],
            [32, 2, 1],
            [32, 2, 2],
            [32, 2, 2],
            [32, 3, 2],
            # now the validation set.
            [33, 2, 1],
            [33, 2, 1],
            [33, 1, 1],
        ],
        columns=['date_block_num', 'item_id', 'shop_id'])
    sales_df.index += 10

    # index = sales_df.index.tolist()
    # np.random.shuffle(index)

    X = pd.DataFrame(np.random.rand(sales_df.shape[0], 10), index=sales_df.index)
    y = pd.Series(np.random.rand((X.shape[0])), index=X.index)

    expected_train_X = X.loc[sales_df.iloc[:-3].index]
    expected_train_y = y.loc[expected_train_X.index]
    mv = ModelValidator(sales_df, X, y)
    train_X, train_y, val_X, val_y = mv.get_train_val_data()

    assert expected_train_X.equals(train_X)
    assert expected_train_y.equals(train_y)

    expected_nz_val_X = X.iloc[[-3, -1]]
    expected_nz_val_y = y[expected_nz_val_X.index]
    nz_val_y = val_y[val_y != 0]
    nz_val_X = X.loc[nz_val_y.index]

    assert nz_val_X.equals(expected_nz_val_X)
    assert nz_val_y.equals(expected_nz_val_y)

    expected_val_ids = set(itertools.product(mock_get_items(), mock_get_shops()))
    val_ids = set(val_X[['item_id', 'shop_id']].astype('int').apply(tuple, axis=1).values)

    assert val_ids == expected_val_ids, "All possible pairs of shop_id, item_id not taken"


@patch('model_validator.get_shops_in_market', side_effect=mock_get_shops)
@patch('model_validator.get_items_in_market', side_effect=mock_get_items)
def test_get_new_items_shops_info(m1, m2):
    sales_df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 1, 1],
            [32, 1, 2],
            [32, 2, 1],
            [32, 2, 2],
            [32, 2, 2],
            [32, 1, 2],
            # now the validation set.
            [33, 2, 1],
            [33, 3, 1],
            [33, 1, 1],
        ],
        columns=['date_block_num', 'item_id', 'shop_id'])
    sales_df.index += 10

    # index = sales_df.index.tolist()
    # np.random.shuffle(index)

    X = pd.DataFrame(np.random.rand(sales_df.shape[0], 10), index=sales_df.index)
    y = pd.Series(np.random.rand((X.shape[0])), index=X.index)

    mv = ModelValidator(sales_df, X, y)
    output = mv.get_new_items_shops_info()

    assert all(np.array(output['new_items']) == np.array([3]))
    assert len(output['new_shops']) == 0
    assert set(output['new_item_shops']) == set([(3, 1)])
