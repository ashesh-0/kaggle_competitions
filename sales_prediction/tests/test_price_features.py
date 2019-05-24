import numpy as np
import pandas as pd
from price_features import get_price_features, get_dollar_value_features


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
    ]
    index = list(range(len(train_data)))
    np.random.shuffle(index)
    sales_df = pd.DataFrame(train_data, columns=columns, index=index)

    return sales_df


def test_price_features_should_work_for_future_month():
    """
    sales_df has maximum date_block_num of 25, then X_df's entries with date_block_num of 26 should be handled properly.
    """
    sales_df = dummy_data()
    X_df = sales_df.copy()
    new_dbn = X_df.date_block_num.max() + 1
    idx1 = X_df.index.max() + 1
    X_df.loc[idx1, :] = [new_dbn, 1, 0, 180, 5]

    idx2 = X_df.index.max() + 1
    X_df.loc[idx2, :] = [new_dbn, 2, 0, 180, 5]

    new_X_df = get_price_features(sales_df, X_df)
    assert new_X_df.loc[idx1, 'avg_item_price'] == 130
    assert new_X_df.loc[idx1, 'std_item_price'] == np.float32(np.std([100, 160], ddof=1))
    assert new_X_df.loc[idx1, 'last_item_price'] == 130

    assert new_X_df.loc[idx2, 'avg_item_price'] == 130
    assert new_X_df.loc[idx2, 'std_item_price'] == np.float32(np.std([100, 160], ddof=1))
    assert new_X_df.loc[idx2, 'last_item_price'] == 100


def test_dollar_value_features():
    sales_df = dummy_data()
    sales_df['item_price'] = (sales_df['item_price'] / sales_df['item_cnt_day']).fillna(0)
    X_df = sales_df.copy()
    X_df['f1'] = np.random.rand(X_df.shape[0])
    X_df['f2'] = np.random.rand(X_df.shape[0])

    # this ensures that we ignore the X_df's price data.
    X_df['item_price'] = np.random.rand(X_df.shape[0])
    orig_X_df_index = X_df.index

    new_X_df = get_dollar_value_features(sales_df, X_df)
    _test_price_features(sales_df, X_df, new_X_df, orig_X_df_index, 'dollar_value')


def test_price_features():
    sales_df = dummy_data()
    X_df = sales_df.copy()
    X_df['f1'] = np.random.rand(X_df.shape[0])
    X_df['f2'] = np.random.rand(X_df.shape[0])

    # this ensures that we ignore the X_df's price data.
    X_df['item_price'] = np.random.rand(X_df.shape[0])
    orig_X_df_index = X_df.index

    new_X_df = get_price_features(sales_df, X_df)
    _test_price_features(sales_df, X_df, new_X_df, orig_X_df_index, 'item_price')


def _test_price_features(sales_df, X_df, new_X_df, orig_X_df_index, price_col):

    assert not new_X_df.isna().any().any()
    assert not np.isinfinite(new_X_df.values).any()
    # index is same.
    assert new_X_df.index.equals(X_df.index)
    assert orig_X_df_index.equals(new_X_df.index)

    cols = X_df.columns.tolist()
    # old data is same.
    assert new_X_df[cols].equals(X_df[cols])

    std_col = 'std_' + price_col
    avg_col = 'avg_' + price_col
    last_col = 'last_' + price_col

    # first month has useless values.
    assert all(new_X_df.iloc[:5][std_col].unique() == -10)
    assert all(new_X_df.iloc[:5][avg_col].unique() == -10)
    assert all(new_X_df.iloc[:5][last_col].unique() == -10)

    # ignoring zero sales entries.
    assert new_X_df.iloc[5][std_col] == 0
    assert new_X_df.iloc[5][avg_col] == 150
    assert new_X_df.iloc[5][last_col] == 150

    assert new_X_df.iloc[6][std_col] == np.float32(np.std([200, 160], ddof=1))
    assert new_X_df.iloc[6][avg_col] == 180
    assert new_X_df.iloc[6][last_col] == 160

    # ignoring zero sales entries.
    assert new_X_df.iloc[7][std_col] == 0
    assert new_X_df.iloc[7][avg_col] == 150
    assert new_X_df.iloc[7][last_col] == 150

    assert new_X_df.iloc[8][std_col] == 0
    assert new_X_df.iloc[8][avg_col] == 300
    assert new_X_df.iloc[8][last_col] == 300

    assert new_X_df.iloc[9][std_col] == np.float32(np.std([200, 160], ddof=1))
    assert new_X_df.iloc[9][avg_col] == 180
    assert new_X_df.iloc[9][last_col] == 180
