from datetime import datetime
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm_notebook
from numeric_utils import compute_concurrently
from numeric_rolling_features import ndays_features
from numeric_monthly_features import MonthlyFeatures
from numeric_overall_features import OverallFeatures


def get_y(sales_df):
    basic_preprocessing(sales_df)
    # now, it is sorted by shop_id, item_id and date.
    # Also, shop_item_group value changes when either the shop or the item changes.

    rev_sales_df = sales_df[::-1]
    y_values = np.zeros(rev_sales_df.shape[0])
    inverted_data = rev_sales_df[['days', 'item_cnt_day', 'shop_item_group']]
    # inverted_index = rev_sales_df.index.tolist()

    tail_index_position = 0
    head_index_position = -1
    tail_index = inverted_data.index[tail_index_position]
    tail_day = inverted_data.iloc[0]['days']
    cur_group = inverted_data.iloc[0]['shop_item_group']
    running_sum = 0
    for head_index in tqdm_notebook(rev_sales_df.index):
        head_group = inverted_data.at[head_index, 'shop_item_group']
        head_index_position += 1

        if head_group != cur_group:
            cur_group = head_group
            tail_index = head_index
            tail_index_position = head_index_position
            running_sum = inverted_data.at[head_index, 'item_cnt_day']
            tail_day = inverted_data.at[head_index, 'days']
        else:
            running_sum += inverted_data.at[head_index, 'item_cnt_day']

        head_day = inverted_data.at[head_index, 'days']
        while tail_day - head_day > 30:
            running_sum -= inverted_data.at[tail_index, 'item_cnt_day']
            tail_index_position += 1
            tail_index = inverted_data.index[tail_index_position]
            tail_day = inverted_data.at[tail_index, 'days']

        y_values[head_index_position] = running_sum

    y_df = pd.Series(y_values, index=rev_sales_df.index)

    return y_df.loc[sales_df.index]


def date_preprocessing(sales_df):
    start_date = datetime(2013, 1, 1)
    if 'date_f' not in sales_df:
        sales_df['date_f'] = pd.to_datetime(sales_df.date, format='%d.%m.%Y')
        sales_df['month'] = sales_df.date_f.apply(lambda x: x.month).astype('uint8')
        sales_df['year'] = sales_df.date_f.apply(lambda x: x.year).astype('uint16')
        sales_df['days'] = (sales_df['date_f'] - start_date).apply(lambda x: x.days)


def basic_preprocessing(sales_df):
    date_preprocessing(sales_df)
    sales_df.sort_values(['shop_id', 'item_id', 'date_f'], inplace=True)
    shop_id_changed = sales_df.shop_id.diff() != 0
    item_id_changed = sales_df.item_id.diff() != 0
    ids_changed = shop_id_changed | item_id_changed
    sales_df['shop_item_group'] = ids_changed.cumsum()


def get_numeric_rolling_feature_df(sales_df, process_count=4):
    basic_preprocessing(sales_df)

    sales_df['log_p'] = np.log(sales_df['item_price'])
    quantiles = [0.25, 0.5, 0.75, 0.9]
    # price_1M_features_args = [(ndays_features, (sales_df, 'log_p', 30, quantiles), {})]

    # sales_half_M_features_args = [(ndays_features, (sales_df, 'item_cnt_day', 15, quantiles), {})]
    sales_1M_features_args = [(ndays_features, (sales_df, 'item_cnt_day', 30, quantiles), {})]
    # sales_1_half_M_features_args = [(ndays_features, (sales_df, 'item_cnt_day', 45, quantiles), {})]
    sales_2M_features_args = [(ndays_features, (sales_df, 'item_cnt_day', 60, quantiles), {})]
    sales_3M_features_args = [(ndays_features, (sales_df, 'item_cnt_day', 90, quantiles), {})]

    args = sales_1M_features_args
    args += sales_2M_features_args
    args += sales_3M_features_args

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

        self._monthly_features = MonthlyFeatures(self._sales_df)
        self._overall_features = OverallFeatures(self._sales_df, self._items_df)

    def _add_features(self, output_df: pd.DataFrame, feature_id_list: List[str], function):
        """
        Args:
            output_df: Newly added features will be added on this dataframe and returned.
            feature_id_list: list of columns of output_df which is to be treated as id for feature generation. For
                    example, for creating features for item_ids every month, we need to pass ['item_id', 'month'] as
                    feature_id_list
            function: which function to be applied to each unique value set of feature_id_list

        """
        unique_feature_id_df = output_df[feature_id_list
                                         + ['index']].groupby(feature_id_list)['index'].count().reset_index()
        features_df = unique_feature_id_df.apply(function, axis=1).astype('float32')
        for col_name in feature_id_list:
            features_df[col_name] = unique_feature_id_df[col_name]

        output_df = pd.merge(output_df, features_df, how='left', on=feature_id_list)
        print(' Numeric features on {} is complete'.format(feature_id_list))
        return output_df

    def get(self, sales_df):
        # assert sales_df[sales_df.item_id.isin([83, 173])].empty

        df = get_numeric_rolling_feature_df(sales_df)

        df['price_category'] = np.log1p(sales_df['item_price']).astype(int)
        df['price_sub_category'] = (np.log1p(sales_df['item_price']) - df['price_category']).fillna(0)
        amount = sales_df['item_price'] * sales_df['item_cnt_day']
        amount.loc[amount < 0] = 0

        df['log_dollar_value'] = np.log1p(amount)

        # assert df[df.item_id.isin([83, 173])].empty
        print('Numeric numeric rolling feature computation is complete.')
        df.reset_index(inplace=True)

        df = self._add_features(df, ['shop_id', 'month'], self._monthly_features.shop_features)
        df = self._add_features(df, ['item_category_id', 'month'], self._monthly_features.category_features)
        df = self._add_features(df, ['month'], self._monthly_features.month_features)
        df = self._add_features(df, ['item_id', 'month'], self._monthly_features.item_features)
        print('Monthly numeric feature computation is complete')

        # Overall features
        df = self._add_features(df, ['item_id'], self._overall_features.item_features)
        df = self._add_features(df, ['shop_id'], self._overall_features.shop_features)
        df = self._add_features(df, ['item_category_id'], self._overall_features.category_features)

        df.set_index('index', inplace=True)

        print("Overall numeric feature computation is complete.")

        return df
