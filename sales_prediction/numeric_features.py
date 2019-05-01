import numpy as np
import pandas as pd
from numeric_utils import compute_concurrently
from numeric_rolling_features import ndays_features
from numeric_monthly_features import MonthlyFeatures
from numeric_overall_features import OverallFeatures


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


def date_preprocessing(sales_df):
    if 'date_f' not in sales_df:
        sales_df['date_f'] = pd.to_datetime(sales_df.date, format='%d.%m.%Y')
        sales_df['month'] = sales_df.date_f.apply(lambda x: x.month)
        sales_df['year'] = sales_df.date_f.apply(lambda x: x.year)


def basic_preprocessing(sales_df):
    date_preprocessing(sales_df)
    sales_df.sort_values(['shop_id', 'item_id', 'date_f'], inplace=True)

    if 'shop_item_group' not in sales_df:
        shop_id_changed = sales_df.shop_id.diff() != 0
        item_id_changed = sales_df.item_id.diff() != 0
        ids_changed = shop_id_changed | item_id_changed
        sales_df['shop_item_group'] = ids_changed.cumsum()


def get_numeric_rolling_feature_df(sales_df, process_count=4):
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
    df['item_category_id'] = sales_df['item_category_id']

    # std on first date is NaN
    return df.dropna()


class NumericFeatures:
    def __init__(self, sales_df, items_df):
        self._sales_df = sales_df
        self._items_df = items_df

        basic_preprocessing(sales_df)

        self._monthly_features = MonthlyFeatures(self._sales_df, self._items_df)
        self._overall_features = OverallFeatures(self._sales_df, self._items_df)

    def get(self, sales_df):
        df = get_numeric_rolling_feature_df(sales_df)
        print('Numeric numeric rolling feature computation is complete.')

        item_m_df = df[['item_id', 'month']].apply(self._monthly_features.item_features, axis=1)
        shop_m_df = df[['shop_id', 'month']].apply(self._monthly_features.shop_features, axis=1)
        categ_m_df = df[['item_category_id', 'month']].apply(self._monthly_features.category_features, axis=1)
        m_df = df['month'].apply(self._monthly_features.month_features)
        df = pd.concat([df, item_m_df, shop_m_df, categ_m_df, m_df], axis=1)
        del item_m_df, shop_m_df, categ_m_df, m_df
        print('Monthly numeric feature computation is complete')

        # Overall features
        item_o_df = df['item_id'].apply(self._overall_features.item_features)
        shop_o_df = df['shop_id'].apply(self._overall_features.shop_features)
        categ_o_df = df['item_category_id'].apply(self._overall_features.category_features)
        print(df)
        print(item_o_df)
        print(shop_o_df)
        print(categ_o_df)
        df = pd.concat([df, item_o_df, shop_o_df, categ_o_df], axis=1)
        print("Overall numeric feature computation is complete.")
        return df
