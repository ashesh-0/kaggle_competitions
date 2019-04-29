import numpy as np
import pandas as pd
from multiprocessing import Pool


def ndays_features(sales_df, col_name, num_days):
    groupby_obj = sales_df.groupby('shop_item_group')[col_name].rolling(num_days, min_periods=1)
    fmt = '{}_{}'.format(col_name, num_days) + 'day_{}'
    fns = [
        (groupby_obj.mean, (), fmt.format('mean')),
        (groupby_obj.std, (), fmt.format('std')),
        (groupby_obj.min, (), fmt.format('min')),
        (groupby_obj.max, (), fmt.format('max')),
        (groupby_obj.quantile, (0.1, ), fmt.format('qt_' + str(10))),
        (groupby_obj.quantile, (0.25, ), fmt.format('qt_' + str(25))),
        (groupby_obj.quantile, (0.5, ), fmt.format('qt_' + str(50))),
        (groupby_obj.quantile, (0.75, ), fmt.format('qt_' + str(75))),
        (groupby_obj.quantile, (0.95, ), fmt.format('qt_' + str(95))),
    ]
    return fns


def run(fn_args):
    fn, args, name = fn_args
    print('Starting', args)
    return fn(*args).to_frame(name)


def compute_concurrently(args, process_count=4):

    with Pool(processes=process_count) as pool:
        output = pool.map(run, args)

    df = pd.concat(output, axis=1)
    return df.reset_index(level=0).drop('shop_item_group', axis=1)


def get_y(sales_df):
    basic_preprocessing(sales_df)

    grp = sales_df[::-1].groupby('shop_item_group')['item_cnt_day'].rolling(30, min_periods=4)
    sum_sales_df = grp.mean().dropna()

    # Extrapolate to 30 days.
    sum_sales_df = sum_sales_df * 30

    # Get it in correct order of date.
    sum_sales_df = sum_sales_df.reset_index(level=0).drop('shop_item_group', axis=1)
    sum_sales_df = sum_sales_df.sort_index()

    # We don't want to use data for November.
    filtr = sales_df.loc[sum_sales_df.index]['date_f'] < pd.Timestamp(year=2015, month=11, day=1)
    return sum_sales_df[filtr]


def basic_preprocessing(sales_df):
    if 'date_f' not in sales_df:
        sales_df['date_f'] = pd.to_datetime(sales_df.date, format='%d.%m.%Y')
        sales_df['month'] = sales_df.date_f.apply(lambda x: x.month)
        sales_df['year'] = sales_df.date_f.apply(lambda x: x.year)

    sales_df.sort_values(['shop_id', 'item_id', 'date_f'], inplace=True)

    if 'shop_item_group' not in sales_df:
        shop_id_changed = sales_df.shop_id.diff() != 0
        item_id_changed = sales_df.item_id.diff() != 0
        ids_changed = shop_id_changed | item_id_changed
        sales_df['shop_item_group'] = ids_changed.cumsum()


def get_numeric_X_df(sales_df, process_count=4):
    basic_preprocessing(sales_df)

    sales_df['log_p'] = np.log(sales_df['item_price'])

    price_1M_features_args = ndays_features(sales_df, 'log_p', 30)

    sales_1M_features_args = ndays_features(sales_df, 'item_cnt_day', 30)
    sales_3M_features_args = ndays_features(sales_df, 'item_cnt_day', 90)

    args = price_1M_features_args + sales_1M_features_args + sales_3M_features_args

    df = compute_concurrently(args, process_count=process_count)

    df['month'] = sales_df['month']
    df['year'] = sales_df['year'] - 2014
    df['shop_id'] = sales_df['shop_id']
    df['item_id'] = sales_df['item_id']

    # std on first date is NaN
    return df.dropna()
