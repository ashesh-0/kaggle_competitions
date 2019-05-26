from constants import TEST_LIKE_SALES_FKEY, TEST_LIKE_SALES_FPATH
import pandas as pd
import numpy as np


def zero_padded_str(num):
    num = int(num)
    return str(num) if num >= 10 else '0{}'.format(num)


def date_str(year, month, day):
    return '{day}.{month}.{year}'.format(year=year, month=zero_padded_str(month), day=zero_padded_str(day))


def date_str_from_dbn(date_block_num, day):
    year = int(date_block_num // 12 + 2013)
    month = date_block_num % 12 + 1
    return date_str(year, month, day)


def get_new_item_shop_ids(item_ids, shop_ids, size, existing_item_shop_ids):
    new_item_shop_ids = {}
    num_times = 0
    while len(new_item_shop_ids) == 0 or num_times < 5:
        num_times += 1
        item_ids = np.random.choice(item_ids, size=size)
        shop_ids = np.random.choice(shop_ids, size=size)
        item_shop_ids = item_ids * 100 + shop_ids
        new_item_shop_ids = list(set(item_shop_ids) - set(existing_item_shop_ids))
    return item_shop_ids[np.isin(item_shop_ids, new_item_shop_ids)]


def shrink_train_data(train_df, aggregate_days):
    """
    Args:
        aggregate_days: number of days for which we will create an aggregated single entry for each item_id,shop_id.
        This is the factor by which train_df's size will reduce.
    """
    train_df['date_f'] = pd.to_datetime(train_df['date'], format='%d.%m.%Y')
    train_df['day'] = train_df['date_f'].apply(lambda x: x.day)
    train_df['day_agg'] = train_df['day'] // aggregate_days + 1

    train_df.index.name = 'index'
    train_df = train_df.reset_index()
    output_df = train_df.groupby(['item_id', 'shop_id', 'date_block_num',
                                  'day_agg'])[['item_price', 'item_cnt_day', 'index']].agg({
                                      'item_price': 'mean',
                                      'item_cnt_day': 'sum',
                                      'index': 'last',
                                  })
    output_df = output_df.reset_index().set_index('index')
    output_df['date'] = output_df.apply(lambda row: date_str_from_dbn(row['date_block_num'], row['day_agg']), axis=1)

    return output_df.drop('day_agg', axis=1).sort_index()


def make_train_have_similar_zeroed_entries_as_test(train_df, train_y_zero_count, train_y_nonzero_count,
                                                   test_zero_fraction):
    """
    for every month do the following:
        get list of active products
        get list of active shops
        find how many rows to insert.
        for each row:
            sample a product and a shop, select an available date. add an entry.

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
    # We don't want to add zero rows in the first month as i fear there could be issues.
    date_block_num.remove(0)

    zero_rows_required = test_zero_fraction * train_y_nonzero_count - train_y_zero_count

    final_expected_columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']

    rows_per_dbn = zero_rows_required // len(date_block_num)

    active_item_ids = {}
    data = []
    for dbn in date_block_num:
        month = dbn % 12 + 1
        year = dbn // 12 + 2013
        unique_days = train_df[train_df.date_block_num == dbn].day.unique()
        num_days = len(unique_days)
        print('Days with entries', unique_days, 'Total num days in a month', num_days)

        existing_item_shop_ids = set(train_df[train_df.date_block_num == dbn].item_shop_id.values)
        filtr = (train_df.date_block_num == dbn) | (train_df.date_block_num == dbn - 1)
        cur_df = train_df[filtr]
        median_price = cur_df.groupby('item_id')['item_price'].quantile(0.5)

        active_item_ids = list(set(cur_df.item_id.unique()))
        active_shop_ids = list(set(cur_df.shop_id.unique()))

        data_dbn_df = pd.DataFrame([], columns=final_expected_columns)
        num_times = 0
        while data_dbn_df.shape[0] != rows_per_dbn and num_times < 5:
            num_times += 1
            new_item_shop_ids = get_new_item_shop_ids(active_item_ids, active_shop_ids, int(1.1 * rows_per_dbn),
                                                      existing_item_shop_ids)
            if len(new_item_shop_ids) == 0:
                print('No new item,shop could be found. continuing to next month')
                break

            # print('got new item_shops')
            cur_zero_df = pd.DataFrame(new_item_shop_ids.reshape(-1, 1), columns=['item_shop_id'])
            data_dbn_df = pd.concat([data_dbn_df, cur_zero_df], axis=0, sort=True)
            data_dbn_df['num_occurances'] = data_dbn_df.groupby('item_shop_id').cumcount() + 1

            # print('Num occurances created')
            data_dbn_df = data_dbn_df[data_dbn_df['num_occurances'] <= num_days]
            data_dbn_df = data_dbn_df.iloc[:rows_per_dbn]
            print('DBN:', dbn, round(data_dbn_df.shape[0] / rows_per_dbn * 100, 2), '% done')

        data_dbn_df['item_id'] = data_dbn_df['item_shop_id'] // 100
        data_dbn_df['shop_id'] = data_dbn_df['item_shop_id'] % 100

        data_dbn_df['date'] = data_dbn_df['num_occurances'].apply(lambda x: date_str(year, month, x))
        data_dbn_df['date_block_num'] = dbn
        data_dbn_df['item_cnt_day'] = 0
        data_dbn_df['item_price'] = data_dbn_df['item_id'].map(median_price)
        data_dbn_df.drop(['num_occurances'], axis=1, inplace=True)

        assert set(final_expected_columns).issubset(set(data_dbn_df.columns))
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
    df = pd.read_csv('../input/sales_train.csv')
    df_shrinked = shrink_train_data(df, 32)
    print('Original size:', '{}K'.format(df.shape[0] // 1000))
    print('Shrinked size:', '{}K'.format(df_shrinked.shape[0] // 1000))
    df = df_shrinked

    monthly_sales = df.groupby(['date_block_num', 'item_id', 'shop_id'])['item_cnt_day'].sum()
    zero_fraction = 5
    output_df = make_train_have_similar_zeroed_entries_as_test(df, (monthly_sales > 0).sum(), df.shape[0],
                                                               zero_fraction)

    monthly_sales_after = output_df.groupby(['date_block_num', 'item_id', 'shop_id'])['item_cnt_day'].sum()

    print('Aiming for fraction to be ', zero_fraction)
    print('Original zero fraction',
          monthly_sales[monthly_sales <= 0].shape[0] / monthly_sales[monthly_sales > 0].shape[0])
    print(
        'New Zero fraction',
        monthly_sales_after[monthly_sales_after <= 0].shape[0] / monthly_sales_after[monthly_sales_after > 0].shape[0])

    output_df.to_hdf(TEST_LIKE_SALES_FPATH, TEST_LIKE_SALES_FKEY)
