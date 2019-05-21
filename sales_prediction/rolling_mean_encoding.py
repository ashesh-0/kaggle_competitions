import pandas as pd
import numpy as np


def _rolling_mean_encoding_by_id(combined_sales_df, col_id):
    df = combined_sales_df.groupby([col_id,
                                    'date_block_num'])['item_cnt_day'].mean().to_frame('item_cnt_month').reset_index()

    df.sort_values('date_block_num', inplace=True)

    sum_df = df.groupby(col_id)['item_cnt_month'].cumsum() - df['item_cnt_month']
    count_df = df.groupby(col_id)['item_cnt_month'].cumcount()
    id_encoding = sum_df / count_df
    id_encoding[count_df == 0] = -10
    id_encoding = id_encoding.to_frame(col_id + '_enc')

    assert id_encoding.index.equals(df.index)
    id_encoding['date_block_num'] = df['date_block_num']
    id_encoding[col_id] = df[col_id]
    return id_encoding


def rolling_mean_encoding(combined_sales_df):

    int64 = combined_sales_df[['item_id', 'shop_id', 'item_category_id']].dtypes == np.int64
    int32 = combined_sales_df[['item_id', 'shop_id', 'item_category_id']].dtypes == np.int32
    assert (int32 | int64).all()

    orig_index = combined_sales_df.index
    combined_sales_df.reset_index(inplace=True)

    # item_id
    item_encoding = _rolling_mean_encoding_by_id(combined_sales_df, 'item_id')
    print('Item id encoding completed')

    # category_id
    category_encoding = _rolling_mean_encoding_by_id(combined_sales_df, 'item_category_id')
    print('Category id encoding completed')

    # shop_encoding
    shop_encoding = _rolling_mean_encoding_by_id(combined_sales_df, 'shop_id')
    print('Shop id encoding completed')

    # item shop_encoding
    combined_sales_df['item_shop_id'] = combined_sales_df['item_id'] * 100 + combined_sales_df['shop_id']
    item_shop_encoding = _rolling_mean_encoding_by_id(combined_sales_df, 'item_shop_id')
    print('Item shop id encoding completed')

    # shop category id
    combined_sales_df['shop_category_id'] = combined_sales_df['shop_id'] * 100 + combined_sales_df['item_category_id']
    shop_category_encoding = _rolling_mean_encoding_by_id(combined_sales_df, 'shop_category_id')
    print('Shop category id encoding completed')

    combined_sales_df = pd.merge(combined_sales_df, item_encoding, on=['item_id', 'date_block_num'], how='left')
    combined_sales_df = pd.merge(
        combined_sales_df, category_encoding, on=['item_category_id', 'date_block_num'], how='left')

    combined_sales_df = pd.merge(combined_sales_df, shop_encoding, on=['shop_id', 'date_block_num'], how='left')

    combined_sales_df = pd.merge(
        combined_sales_df, item_shop_encoding, on=['item_shop_id', 'date_block_num'], how='left')

    combined_sales_df = pd.merge(
        combined_sales_df, shop_category_encoding, on=['shop_category_id', 'date_block_num'], how='left')

    combined_sales_df.drop(['item_shop_id', 'shop_category_id'], axis=1, inplace=True)
    combined_sales_df.set_index('index', inplace=True)

    return combined_sales_df.loc[orig_index]
