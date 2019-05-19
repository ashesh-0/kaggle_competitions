import itertools
import numpy as np


def fill_missing_month(df):
    skipped_months = set(range(35)) - set(df.columns.tolist())
    if skipped_months:
        for month in skipped_months:
            df[month] = 0

    df.sort_index(axis=1, inplace=True)
    assert set(range(35)) == set(df.columns.tolist())


def rolling_operation(df):
    feature_df = df.rolling(df.shape[1], min_periods=1).mean()
    # no data leakage.
    feature_df = feature_df.shift(1, axis=1)
    # first column is all nans.
    feature_df[0] = 0
    return feature_df.sort_index()


def rolling_mean_encoding(sales_df):
    # item_id
    df = sales_df.groupby(['item_id', 'date_block_num'])['item_cnt_day'].mean().unstack().fillna(0)
    fill_missing_month(df)
    item_encoding = rolling_operation(df)

    print('Item id encoding completed')

    # category_id
    df = sales_df.groupby(['item_category_id', 'date_block_num'])['item_cnt_day'].mean().unstack().fillna(0)
    fill_missing_month(df)
    category_encoding = rolling_operation(df)

    print('Category id encoding completed')
    # shop_encoding
    df = sales_df.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].mean().unstack().fillna(0)
    fill_missing_month(df)
    shop_encoding = rolling_operation(df)

    print('Shop id encoding completed')

    # shop_encoding
    sales_df['item_shop_id'] = sales_df['item_id'] * 100 + sales_df['shop_id']
    df = sales_df.groupby(['item_shop_id', 'date_block_num'])['item_cnt_day'].mean().unstack().fillna(0)
    fill_missing_month(df)
    item_shop_encoding = rolling_operation(df)
    print('Item shop id encoding completed')

    # shop category id
    sales_df['shop_category_id'] = sales_df['shop_id'] * 100 + sales_df['item_category_id']
    df = sales_df.groupby(['shop_category_id', 'date_block_num'])['item_cnt_day'].mean().unstack().fillna(0)
    fill_missing_month(df)
    shop_category_encoding = rolling_operation(df)
    print('Shop category id encoding completed')

    return {
        'item_id': item_encoding.astype(np.float32),
        'item_category_id': category_encoding.astype(np.float32),
        'shop_id': shop_encoding.astype(np.float32),
        'item_shop_id': item_shop_encoding.astype(np.float32),
        'shop_category_id': shop_category_encoding.astype(np.float32),
    }


def fill_missing_keys(encoding_dict, ids):
    for ids in ids:
        for m in range(34):
            key = ids * 100 + m
            if key in encoding_dict:
                continue
            encoding_dict[key] = 0


def find_unique_ids(shops_df, items_df):
    unique_ids_dict = {}
    unique_ids_dict['item_id'] = items_df.item_id.unique()
    unique_ids_dict['shop_id'] = shops_df.shop_id.unique()
    unique_ids_dict['item_category_id'] = items_df.item_category_id.unique()

    unique_ids_dict['item_shop_id'] = itertools.product(unique_ids_dict['item_id'], unique_ids_dict['shop_id'])
    unique_ids_dict['item_shop_id'] = list(map(lambda x: x[0] * 100 + x[1], unique_ids_dict['item_shop_id']))

    unique_ids_dict['shop_category_id'] = itertools.product(unique_ids_dict['shop_id'],
                                                            unique_ids_dict['item_category_id'])
    unique_ids_dict['shop_category_id'] = list(map(lambda x: x[0] * 100 + x[1], unique_ids_dict['shop_category_id']))
    return unique_ids_dict


def change_encoding_df_to_dict(unique_ids_dict):
    for col in ['item_id', 'shop_id', 'item_category_id', 'item_shop_id', 'shop_category_id']:
        df = mean_encodings.pop(col).stack().to_frame('item_cnt_month').reset_index()
        col_dbn = col + '_dbn'
        df[col_dbn] = df[col] * 100 + df['date_block_num']
        encoding_dict = df.set_index(col_dbn)['item_cnt_month'].to_dict()
        unique_ids = unique_ids_dict[col]
        fill_missing_keys(encoding_dict, unique_ids)
        del df
        mean_encodings[col_dbn] = encoding_dict
        print(col, ' converted to dict')


def add_id_dbn_col(df, col):
    df[col + '_dbn'] = df[col] * 100 + df['date_block_num']


def apply_item_encoding(df):
    add_id_dbn_col(df, 'item_id')
    item_encoding = mean_encodings['item_id_dbn']
    df['item_id_mean_enc'] = df['item_id_dbn'].map(item_encoding).astype(np.float32)
    df.drop('item_id_dbn', axis=1, inplace=True)


def apply_item_category_encoding(df):
    add_id_dbn_col(df, 'item_category_id')
    item_category_encoding = mean_encodings['item_category_id_dbn']
    df['item_category_id_mean_enc'] = df['item_category_id_dbn'].map(item_category_encoding).astype(np.float32)
    df.drop('item_category_id_dbn', axis=1, inplace=True)


def apply_shop_encoding(df):
    add_id_dbn_col(df, 'shop_id')
    shop_encoding = mean_encodings['shop_id_dbn']
    df['shop_id_mean_enc'] = df['shop_id_dbn'].map(shop_encoding).astype(np.float32)
    df.drop('shop_id_dbn', axis=1, inplace=True)


def apply_item_shop_encoding(df):
    item_shop_encoding = mean_encodings['item_shop_id_dbn']
    df['item_shop_id'] = df['item_id'] * 100 + df['shop_id']
    add_id_dbn_col(df, 'item_shop_id')

    df['item_shop_id_mean_enc'] = df['item_shop_id_dbn'].map(item_shop_encoding).astype(np.float32)
    df.drop(['item_shop_id', 'item_shop_id_dbn'], axis=1, inplace=True)


def apply_shop_category_encoding(df):
    shop_category_encoding = mean_encodings['shop_category_id_dbn']
    df['shop_category_id'] = df['shop_id'] * 100 + df['item_category_id']
    add_id_dbn_col(df, 'shop_category_id')
    df['shop_category_id_mean_enc'] = df['shop_category_id_dbn'].map(shop_category_encoding).astype(np.float32)
    df.drop(['shop_category_id', 'shop_category_id_dbn'], axis=1, inplace=True)


def apply_mean_encoding(df):
    apply_item_encoding(df)
    print("item encoding applied")
    apply_item_category_encoding(df)
    print("category encoding applied")
    apply_shop_encoding(df)
    print("shop encoding applied")
    apply_item_shop_encoding(df)
    print("item shop encoding applied")
    apply_shop_category_encoding(df)
    print("shop category encoding applied")


mean_encodings = rolling_mean_encoding(sales_df)
unique_ids_dict = find_unique_ids(shops_df, items_df)
change_encoding_df_to_dict(mean_encodings)
