import numpy as np

from numeric_utils import compute_concurrently
from numeric_rolling_features import nmonths_features
from price_features import get_price_features


def get_y(sales_df):
    """
    Note that we are not using this month's data in features. we are using previous month's data.
    """
    return sales_df['item_cnt_day']


def date_preprocessing(sales_df):
    sales_df['month'] = sales_df['date_block_num'] % 12 + 1
    sales_df['year'] = (sales_df['date_block_num'] // 12 + 2013).astype(np.int16)


def basic_preprocessing(sales_df):
    date_preprocessing(sales_df)
    sales_df.sort_values(['shop_id', 'item_id', 'date_block_num'], inplace=True)
    shop_id_changed = sales_df.shop_id.diff() != 0
    item_id_changed = sales_df.item_id.diff() != 0
    ids_changed = shop_id_changed | item_id_changed
    sales_df['shop_item_group'] = ids_changed.cumsum()


def get_numeric_rolling_feature_df(sales_df, process_count=4):
    basic_preprocessing(sales_df)

    sales_df['log_p'] = np.log(sales_df['item_price'])
    quantiles = [0.25, 0.5, 0.75, 0.9]

    sales_1M_features_args = [(nmonths_features, (sales_df, 'item_cnt_day', 1, quantiles), {})]
    sales_2M_features_args = [(nmonths_features, (sales_df, 'item_cnt_day', 2, quantiles), {})]
    sales_4M_features_args = [(nmonths_features, (sales_df, 'item_cnt_day', 4, quantiles), {})]

    args = sales_1M_features_args
    args += sales_2M_features_args
    args += sales_4M_features_args

    df = compute_concurrently(args, process_count=process_count).astype('float32')

    df['month'] = sales_df['month'].astype('uint8')
    df['year'] = sales_df['year'] - 2013
    df['shop_id'] = sales_df['shop_id'].astype('uint8')
    df['item_id'] = sales_df['item_id'].astype('uint16')
    df['item_category_id'] = sales_df['item_category_id'].astype('uint8')

    # std on first date is NaN
    print('Number of nan elements', df.isna().sum().sum())
    return df


class NumericFeatures:
    def __init__(self, sales_df, items_df):
        self._sales_df = sales_df
        self._items_df = items_df

        basic_preprocessing(sales_df)

    def get(self, sales_df):
        # assert sales_df[sales_df.item_id.isin([83, 173])].empty

        df = get_numeric_rolling_feature_df(sales_df)
        print('Numeric rolling feature added.')

        # price features
        df['date_block_num'] = sales_df['date_block_num'].astype('uint8')
        df = get_price_features(sales_df, df)
        df.drop('date_block_num', axis=1, inplace=True)
        print('Price features added')

        return df
