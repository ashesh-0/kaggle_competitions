import numpy as np
import pandas as pd

INVALID_VALUE = -10


def get_price_features(sales_df, X_df):
    """
    sales_df is monthly
    """
    # use original sales data.
    sales_df = sales_df[sales_df.item_cnt_day > 0].copy()
    sales_df.loc[sales_df.item_price < 0, 'item_price'] = 0

    grp = sales_df.groupby(['item_id', 'date_block_num'])['item_price']
    # std for 1 entry should be 0. std for 0 entry should be -10
    std = grp.std().fillna(0).unstack().sort_index(axis=1).fillna(
        method='ffill', axis=1).shift(
            1, axis=1).fillna(INVALID_VALUE)

    avg_price = grp.mean().unstack().sort_index(axis=1).fillna(
        method='ffill', axis=1).shift(
            1, axis=1).fillna(INVALID_VALUE)

    avg_price = avg_price.stack().to_frame('avg_price').reset_index()
    std = std.stack().to_frame('price_std').reset_index()

    last_price_df = sales_df[['item_id', 'shop_id', 'date_block_num', 'item_price']].copy()
    last_price_df['date_block_num'] += 1
    last_price_df.rename({'item_price': 'last_item_price'}, inplace=True, axis=1)

    # index
    X_df = X_df.reset_index()

    # item_id price
    X_df = pd.merge(X_df, avg_price, on=['item_id', 'date_block_num'], how='left')
    X_df['avg_price'] = X_df['avg_price'].fillna(INVALID_VALUE)

    # shop_id item_id coupled price
    X_df = pd.merge(X_df, last_price_df, on=['item_id', 'shop_id', 'date_block_num'], how='left')

    X_df['last_item_price'] = X_df['last_item_price'].fillna(X_df['avg_price'])

    # stdev
    X_df = pd.merge(X_df, std, on=['item_id', 'date_block_num'], how='left')
    X_df['price_std'] = X_df['price_std'].fillna(INVALID_VALUE)

    X_df.set_index('index', inplace=True)
    X_df[['price_std', 'last_item_price', 'avg_price']] = X_df[['price_std', 'last_item_price', 'avg_price']].astype(
        np.float32)

    price_category = np.log1p(X_df['last_item_price'] - INVALID_VALUE)
    X_df['price_category'] = price_category.astype(int)
    X_df['price_sub_category'] = (price_category - X_df['price_category']).astype(np.float32)

    avg_price_category = np.log1p(X_df['avg_price'] - INVALID_VALUE)
    X_df['avg_price_category'] = avg_price_category.astype(int)
    X_df['avg_price_sub_category'] = (avg_price_category - X_df['price_category']).astype(np.float32)

    return X_df
