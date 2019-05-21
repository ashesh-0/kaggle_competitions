import numpy as np


def _rolling_mean_encoding_by_id(combined_sales_df, col_id):
    sum_df = combined_sales_df.groupby(col_id)['item_cnt_day'].cumsum() - combined_sales_df['item_cnt_day']
    count_df = combined_sales_df.groupby(col_id)['item_cnt_day'].cumcount()
    id_encoding = sum_df / count_df
    id_encoding[count_df == 0] = -10
    return id_encoding


def rolling_mean_encoding(combined_sales_df):

    int64 = combined_sales_df[['item_id', 'shop_id', 'item_category_id']].dtypes == np.int64
    int32 = combined_sales_df[['item_id', 'shop_id', 'item_category_id']].dtypes == np.int32
    assert (int32 | int64).all()

    combined_sales_df.reset_index(inplace=True)
    combined_sales_df.sort_values('date_block_num', inplace=True)

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

    combined_sales_df.drop(['shop_category_id', 'item_shop_id'], axis=1, inplace=True)

    combined_sales_df['item_id_enc'] = item_encoding.astype(np.float32).loc[combined_sales_df.index]
    combined_sales_df['item_category_id_enc'] = category_encoding.astype(np.float32).loc[combined_sales_df.index]
    combined_sales_df['shop_id_enc'] = shop_encoding.astype(np.float32).loc[combined_sales_df.index]
    combined_sales_df['item_shop_id_enc'] = item_shop_encoding.astype(np.float32).loc[combined_sales_df.index]
    combined_sales_df['shop_category_id_enc'] = shop_category_encoding.astype(np.float32).loc[combined_sales_df.index]

    combined_sales_df.sort_index(inplace=True)
    combined_sales_df.set_index('index', inplace=True)
