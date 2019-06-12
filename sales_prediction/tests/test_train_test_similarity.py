import pandas as pd
import numpy as np
from train_test_similarity import make_train_have_similar_zeroed_entries_as_test, get_monthly_sales


def dummy_sales():
    columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']

    # Train
    data = [
        ['03.11.2013', 10, 0, 0, 10, 2.0],
        ['02.11.2013', 10, 0, 0, 15, 1.0],
        ['03.11.2013', 10, 0, 1, 100, 1.0],
        ['06.11.2013', 10, 1, 1, 100, 1.0],
        ['03.12.2013', 11, 0, 0, 20, 2.0],
        ['03.12.2013', 11, 0, 1, 100, 1.0],
        ['03.12.2013', 11, 0, 2, 1000, 1.0],
        ['03.12.2013', 11, 1, 0, 20, 5.0],
        ['03.12.2013', 11, 1, 1, 100, 2.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    sales_df['item_id'] = sales_df['item_id'].astype(np.int32)
    sales_df['shop_id'] = sales_df['shop_id'].astype(np.int32)
    return sales_df


def test_get_monthly_sales():
    sales_df = dummy_sales()
    dtypes = sales_df.dtypes
    monthly_sales = get_monthly_sales(sales_df)
    monthly_sales = monthly_sales.astype(dtypes)

    cols = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
    assert monthly_sales.iloc[1:][cols].equals(sales_df.iloc[2:][cols])

    cols2 = ['date_block_num', 'shop_id', 'item_id']
    assert all(monthly_sales.iloc[0][cols2] == sales_df[cols2].iloc[0])
    assert all(monthly_sales.iloc[0][cols2] == sales_df[cols2].iloc[1])
    assert monthly_sales.iloc[0]['item_cnt_day'] == 3


def test_make_train_have_similar_zeroed_entries_as_test():
    sales_df = dummy_sales()
    monthly_sales = get_monthly_sales(sales_df)
    output_df = make_train_have_similar_zeroed_entries_as_test(monthly_sales)
    assert output_df[output_df.date_block_num == 10].shape[0] == 4
    assert all(output_df[output_df.date_block_num == 10].groupby('item_id')['shop_id'].count().unique() == [2])
    assert all(output_df[output_df.date_block_num == 10].groupby('shop_id')['item_id'].count().unique() == [2])

    assert output_df[output_df.date_block_num == 11].shape[0] == 6
    assert all(output_df[output_df.date_block_num == 11].groupby('item_id')['shop_id'].count().unique() == [2])
    assert all(output_df[output_df.date_block_num == 11].groupby('shop_id')['item_id'].count().unique() == [3])

    # original data is there.
    assert output_df.loc[monthly_sales.index].equals(monthly_sales.astype(output_df.dtypes))
    # new data is zero sales.
    assert all(output_df[~output_df.index.isin(monthly_sales.index)]['item_cnt_day'].unique() == [0])
    # price is correctly set
    fil = (output_df.date_block_num == 10) & (output_df.item_id == 0) & (output_df.shop_id == 1)
    assert all(output_df[fil]['item_price'] == [np.quantile([10, 15], 0.5)])
