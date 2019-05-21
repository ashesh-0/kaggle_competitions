import numpy as np
import pandas as pd
from rolling_mean_encoding import rolling_mean_encoding


def dummy_data():
    columns = ['month', 'date_block_num', 'shop_id', 'item_id', 'item_category_id', 'item_cnt_day']

    # Train
    train_data = [
        [1, 13, 0, 0, 0, 1],
        [1, 13, 0, 1, 0, 2],
        [1, 13, 0, 2, 1, 3],
        [1, 13, 1, 0, 0, 4],
        [2, 14, 1, 1, 0, 5],
        [2, 14, 2, 0, 0, 6],
        [2, 14, 3, 2, 1, 7],
        [3, 15, 3, 2, 1, 3],
    ]
    index = list(range(len(train_data)))
    np.random.shuffle(index)
    sales_df = pd.DataFrame(train_data, columns=columns, index=index)

    return sales_df


def test_rolling_mean_encoding_does_not_change_index_ordering_and_has_correct_values():
    sales_df = dummy_data()
    orig_index = sales_df.index
    sales_df.sort_index(inplace=True)
    new_index = sales_df.index

    sales_df = rolling_mean_encoding(sales_df)

    assert sales_df.index.equals(new_index)
    sales_df = sales_df.loc[orig_index]

    orig_columns = sales_df.columns.tolist()
    assert set(sales_df.columns) == set(
        orig_columns +
        ['item_id_enc', 'shop_id_enc', 'item_category_id_enc', 'item_shop_id_enc', 'shop_category_id_enc'])

    # -10 is for first time values.
    assert max(abs(sales_df['item_id_enc'].values - [-10, -10, -10, -10, 2 / 1, 5 / 2, 3 / 1, 10 / 2])) < 1e-5

    # Monthly numbers
    #  0 => 2
    #  1 => 4
    #  2 => _, 6
    #  3 => _, 7, 3
    assert max(abs(sales_df['shop_id_enc'].values - [-10, -10, -10, -10, 4 / 1, -10, -10, 7])) < 1e-5

    assert max(abs(sales_df['item_category_id_enc'].values - [-10, -10, -10, -10, 7 / 3, 7 / 3, 3 / 1, 10 / 2])) < 1e-5

    assert max(abs(sales_df['item_shop_id_enc'].values - [-10, -10, -10, -10, -10, -10, -10, 7 / 1])) < 1e-5

    assert max(abs(sales_df['shop_category_id_enc'].values - [-10, -10, -10, -10, 4 / 1, -10, -10, 7 / 1])) < 1e-5
