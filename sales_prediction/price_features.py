import numpy as np
import pandas as pd

INVALID_VALUE = -10


def get_price_features(sales_df, X_df):
    return _get_price_features(sales_df, X_df, 'item_price')


def get_dollar_value_features(sales_df, X_df):
    sales_df['dollar_value'] = (sales_df['item_price'] * sales_df['item_cnt_day']).astype(np.float32)
    output_df = _get_price_features(sales_df, X_df, 'dollar_value')
    sales_df.drop('dollar_value', axis=1, inplace=True)
    return output_df


def _get_price_features(sales_df, X_df, price_col):
    """
    sales_df is monthly
    """
    # use original sales data.
    msg = 'X_df has >1 recent months data. To speed up the process, we are just handling 1 month into the future case'
    assert X_df.date_block_num.max() <= sales_df.date_block_num.max() + 1, msg

    sales_df = sales_df[sales_df.item_cnt_day > 0].copy()
    sales_df.loc[sales_df[price_col] < 0, price_col] = 0

    grp = sales_df.groupby(['item_id', 'date_block_num'])[price_col]
    # std for 1 entry should be 0. std for 0 entry should be -10
    std = grp.std().fillna(0).unstack()
    std[sales_df.date_block_num.max() + 1] = 0
    std = std.sort_index(axis=1).fillna(method='ffill', axis=1).shift(1, axis=1).fillna(INVALID_VALUE)
    std_col = 'std_{}'.format(price_col)
    std = std.stack().to_frame(std_col).reset_index()

    avg_price = grp.mean().unstack()
    avg_price[sales_df.date_block_num.max() + 1] = 0
    avg_price = avg_price.sort_index(axis=1).fillna(method='ffill', axis=1).shift(1, axis=1).fillna(INVALID_VALUE)
    avg_col = 'avg_{}'.format(price_col)
    avg_price = avg_price.stack().to_frame(avg_col).reset_index()

    last_price_df = sales_df[['item_id', 'shop_id', 'date_block_num', price_col]].copy()
    last_price_df['date_block_num'] += 1
    last_pr_col = 'last_{}'.format(price_col)
    last_price_df.rename({price_col: last_pr_col}, inplace=True, axis=1)

    # index
    X_df = X_df.reset_index()

    # item_id price
    X_df = pd.merge(X_df, avg_price, on=['item_id', 'date_block_num'], how='left')
    X_df[avg_col] = X_df[avg_col].fillna(INVALID_VALUE)

    # shop_id item_id coupled price
    X_df = pd.merge(X_df, last_price_df, on=['item_id', 'shop_id', 'date_block_num'], how='left')

    X_df[last_pr_col] = X_df[last_pr_col].fillna(X_df[avg_col])

    # stdev
    X_df = pd.merge(X_df, std, on=['item_id', 'date_block_num'], how='left')
    X_df[std_col] = X_df[std_col].fillna(INVALID_VALUE)

    X_df.set_index('index', inplace=True)
    X_df[[std_col, last_pr_col, avg_col]] = X_df[[std_col, last_pr_col, avg_col]].astype(np.float32)

    price_category = np.log1p(X_df[last_pr_col] - INVALID_VALUE)

    # CATEGORY FEATURES.
    categ_col = 'category_{}'.format(price_col)
    subcateg_col = 'sub_category_{}'.format(price_col)

    avg_categ_col = 'avg_category_{}'.format(price_col)
    avg_subcateg_col = 'avg_sub_category_{}'.format(price_col)

    X_df[categ_col] = price_category.astype(int)
    X_df[subcateg_col] = (price_category - X_df[categ_col]).astype(np.float32)

    avg_price_category = np.log1p(X_df[avg_col] - INVALID_VALUE)
    X_df[avg_categ_col] = avg_price_category.astype(int)
    X_df[avg_subcateg_col] = (avg_price_category - X_df[categ_col]).astype(np.float32)

    return X_df
