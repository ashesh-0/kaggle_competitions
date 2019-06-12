"""
Aim of the functions defined here is to make train data similar to test data. In test data, for each month, every
possible pair of shop_id and item_id is present. This script does the same to train data. price data is filled with
median price.
"""
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook


def zero_padded_str(num):
    num = int(num)
    return str(num) if num >= 10 else '0{}'.format(num)


def date_str(year, month, day):
    return '{day}.{month}.{year}'.format(year=year, month=zero_padded_str(month), day=zero_padded_str(day))


def date_str_from_dbn(date_block_num, day):
    year = int(date_block_num // 12 + 2013)
    month = date_block_num % 12 + 1
    return date_str(year, month, day)


def get_monthly_sales(train_df):
    """
    Make it monthly sales.
    """
    train_df.index.name = 'index'
    train_df = train_df.reset_index()
    output_df = train_df.groupby(['item_id', 'shop_id', 'date_block_num'])[['item_price', 'item_cnt_day',
                                                                            'index']].agg({
                                                                                'item_price': 'mean',
                                                                                'item_cnt_day': 'sum',
                                                                                'index': 'last',
                                                                            })
    output_df = output_df.reset_index().set_index('index')
    output_df['date'] = output_df.apply(lambda row: date_str_from_dbn(row['date_block_num'], 1), axis=1)

    return output_df.sort_index()


def make_train_have_similar_zeroed_entries_as_test(train_df):
    """
    for every month do the following:
        get list of active products
        get list of active shops
        Insert rows for all missing (item_id,shop_id) pair.
        ensure that index of these zeroed elements are greater than max index of train. concatenate and save.
    """
    dtypes = {
        'date_block_num': np.uint8,
        'shop_id': np.int32,
        'item_id': np.int32,
        'item_price': np.float32,
        'item_cnt_day': np.float32,
    }

    train_df['item_shop_id'] = train_df['item_id'] * 100 + train_df['shop_id']
    train_df['date_f'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
    train_df['day'] = train_df['date_f'].apply(lambda x: x.day)

    date_block_num = train_df.date_block_num.unique().tolist()
    date_block_num.sort()

    final_expected_columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']

    data = []
    for dbn in tqdm_notebook(date_block_num):

        month = dbn % 12 + 1
        year = dbn // 12 + 2013
        dbn_train_df = train_df[train_df.date_block_num == dbn]
        median_price = dbn_train_df.groupby('item_id')['item_price'].quantile(0.5)

        item_ids = dbn_train_df.item_id.unique()
        shop_ids = dbn_train_df.shop_id.unique()
        all_item_shop_ids = set([a * 100 + b for (a, b) in itertools.product(item_ids, shop_ids)])
        existing_item_shop_ids = set(dbn_train_df.item_shop_id.values)
        missing_item_shop_ids = all_item_shop_ids - existing_item_shop_ids

        missing_item_ids = [a // 100 for a in missing_item_shop_ids]
        missing_shop_ids = [a % 100 for a in missing_item_shop_ids]
        data_dbn_df = pd.DataFrame([], columns=final_expected_columns)

        data_dbn_df['item_id'] = missing_item_ids
        data_dbn_df['shop_id'] = missing_shop_ids

        data_dbn_df['date'] = date_str(year, month, 1)
        data_dbn_df['date_block_num'] = dbn
        data_dbn_df['item_cnt_day'] = 0
        data_dbn_df['item_price'] = data_dbn_df['item_id'].map(median_price)

        assert data_dbn_df.isna().values.sum() == 0
        data_dbn_df = data_dbn_df.astype(dtypes)
        data.append(data_dbn_df)

    zeros_data_df = pd.concat(data)

    zeros_data_df = zeros_data_df.reset_index()[final_expected_columns]
    zeros_data_df.index += train_df.index.max() + 1

    train_df.drop(['date_f', 'day'], axis=1, inplace=True)
    new_train_df = pd.concat([train_df, zeros_data_df], sort=True).drop('item_shop_id', axis=1)

    assert new_train_df.isna().any().any() == False

    new_train_df = new_train_df.astype(dtypes)

    # ensuring the ordering.
    train_df.drop('item_shop_id', inplace=True, axis=1)
    new_train_df = new_train_df[train_df.columns]

    print(new_train_df.tail())
    return new_train_df


if __name__ == '__main__':
    from constants import TEST_LIKE_SALES_FKEY, TEST_LIKE_SALES_FPATH, SALES_FPATH
    df = pd.read_csv(SALES_FPATH)
    df_shrinked = get_monthly_sales(df)
    print('Original size:', '{}K'.format(df.shape[0] // 1000))
    print('Shrinked size:', '{}K'.format(df_shrinked.shape[0] // 1000))
    df = df_shrinked
    output_df = make_train_have_similar_zeroed_entries_as_test(df)
    print('Final size', output_df.shape)
    output_df.to_hdf(TEST_LIKE_SALES_FPATH, TEST_LIKE_SALES_FKEY)
