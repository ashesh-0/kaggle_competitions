import pandas as pd
import numpy as np
from tqdm import tqdm_notebook


def _rolling_mean_encoding_by_id(combined_sales_df, col_id):
    if col_id == 'item_shop_id':
        assert combined_sales_df.groupby(
            [col_id, 'date_block_num'])['item_cnt_day'].count().shape[0] == combined_sales_df.shape[0]
        df = combined_sales_df[['item_cnt_day', 'item_shop_id', 'date_block_num']].rename(
            {
                'item_cnt_day': 'item_cnt_month'
            }, axis=1)
    else:
        df = combined_sales_df.groupby(
            [col_id, 'date_block_num'])['item_cnt_day'].mean().to_frame('item_cnt_month').reset_index()

    df.sort_values('date_block_num', inplace=True)

    sum_df = df.groupby(col_id)['item_cnt_month'].cumsum() - df['item_cnt_month']
    count_df = df.groupby(col_id)['item_cnt_month'].cumcount()
    id_encoding = sum_df / count_df
    id_encoding[count_df == 0] = -10
    id_encoding = id_encoding.to_frame(col_id + '_enc').astype(np.float32)

    assert id_encoding.index.equals(df.index)
    id_encoding['date_block_num'] = df['date_block_num']
    id_encoding[col_id] = df[col_id]
    return id_encoding


def _rolling_quantile_encoding_by_id(combined_sales_df, col_id, quantile):
    assert col_id != 'item_shop_id'
    df = combined_sales_df.groupby(
        [col_id, 'date_block_num'])['item_cnt_day'].quantile(quantile).to_frame('item_cnt_month').reset_index()

    df.sort_values('date_block_num', inplace=True)

    # Alternate formulation.
    # df.groupby(3)[1].rolling(10, min_periods=1).quantile(0).reset_index(level=0)

    sum_df = df.groupby(col_id)['item_cnt_month'].cumsum() - df['item_cnt_month']
    count_df = df.groupby(col_id)['item_cnt_month'].cumcount()
    id_encoding = sum_df / count_df
    id_encoding[count_df == 0] = -10
    id_encoding = id_encoding.to_frame(col_id + '_qt_{}_enc'.format(quantile)).astype(np.float32)

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

    # item shop_encoding
    combined_sales_df['item_shop_id'] = combined_sales_df['item_id'] * 100 + combined_sales_df['shop_id']
    # shop category id
    combined_sales_df['shop_category_id'] = combined_sales_df['shop_id'] * 100 + combined_sales_df['item_category_id']

    for col_id in ['item_id', 'item_category_id', 'shop_id', 'item_shop_id', 'shop_category_id']:
        # item_id
        encoding = _rolling_mean_encoding_by_id(combined_sales_df, col_id)
        combined_sales_df = pd.merge(combined_sales_df, encoding, on=[col_id, 'date_block_num'], how='left')
        print('{} encoding completed'.format(col_id))

    pbar = tqdm_notebook([0.1, 0.9, 0.95])
    for qt in pbar:
        for col_id in tqdm_notebook(['item_id', 'item_category_id', 'shop_id', 'shop_category_id']):
            pbar.set_description('Running {} Quantile mean encoding for "{}"'.format(qt, col_id))
            # item_id
            encoding = _rolling_quantile_encoding_by_id(combined_sales_df, col_id, qt)
            combined_sales_df = pd.merge(combined_sales_df, encoding, on=[col_id, 'date_block_num'], how='left')
            print('{} encoding completed'.format(col_id))

    combined_sales_df.drop(['item_shop_id', 'shop_category_id'], axis=1, inplace=True)
    combined_sales_df.set_index('index', inplace=True)

    return combined_sales_df.loc[orig_index]
