import numpy as np
import pandas as pd
from price_features import get_price_features


def dummy_data():
    columns = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
    # index is same.
    # ignore X_df price data.
    # take valid price data from sales
    # take mean of price data from sales for non existing.
    # no nans and reset with -10
    train_data = [
        [13, 0, 0, 100, 0],
        [13, 0, 1, 200, 2],
        [13, 1, 1, 160, 5],
        [13, 0, 2, 300, 3],
        [13, 1, 0, 150, 4],
        #
        [14, 0, 0, 160, 1],
        [14, 1, 1, 180, 5],
        [14, 2, 0, 100, 6],
        [14, 3, 2, 250, 7],
        [14, 2, 1, 180, 5],
        #
        [15, 3, 2, 380, 3],
    ]
    index = list(range(len(train_data)))
    np.random.shuffle(index)
    sales_df = pd.DataFrame(train_data, columns=columns, index=index)

    return sales_df


def test_price_features():
    sales_df = dummy_data()
    X_df = sales_df.copy()
    X_df['f1'] = np.random.rand(X_df.shape[0])
    X_df['f2'] = np.random.rand(X_df.shape[0])

    # this ensures that we ignore the X_df's price data.
    X_df['item_price'] = np.random.rand(X_df.shape[0])
    orig_index = X_df.index

    new_X_df = get_price_features(sales_df, X_df)
    # index is same.
    assert new_X_df.index.equals(X_df.index)
    assert orig_index.equals(new_X_df.index)

    cols = X_df.columns.tolist()
    # old data is same.
    assert new_X_df[cols].equals(X_df[cols])

    # first month has useless values.
    assert all(new_X_df.iloc[:5]['price_std'].unique() == -10)
    assert all(new_X_df.iloc[:5]['avg_price'].unique() == -10)
    assert all(new_X_df.iloc[:5]['last_item_price'].unique() == -10)

    # ignoring zero sales entries.
    assert new_X_df.iloc[5]['price_std'] == 0
    assert new_X_df.iloc[5]['avg_price'] == 150
    assert new_X_df.iloc[5]['last_item_price'] == 150

    assert new_X_df.iloc[6]['price_std'] == np.float32(np.std([200, 160], ddof=1))
    assert new_X_df.iloc[6]['avg_price'] == 180
    assert new_X_df.iloc[6]['last_item_price'] == 160

    # ignoring zero sales entries.
    assert new_X_df.iloc[7]['price_std'] == 0
    assert new_X_df.iloc[7]['avg_price'] == 150
    assert new_X_df.iloc[7]['last_item_price'] == 150

    assert new_X_df.iloc[8]['price_std'] == 0
    assert new_X_df.iloc[8]['avg_price'] == 300
    assert new_X_df.iloc[8]['last_item_price'] == 300

    assert new_X_df.iloc[9]['price_std'] == np.float32(np.std([200, 160], ddof=1))
    assert new_X_df.iloc[9]['avg_price'] == 180
    assert new_X_df.iloc[9]['last_item_price'] == 180
