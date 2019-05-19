from sklearn.cluster.bicluster import SpectralBiclustering
import Levenshtein as lev
from typing import List, Tuple
import numpy as np
import pandas as pd

DefaultValue = -1000


class DefaultDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._default = DefaultValue

    def __getitem__(self, idx):
        return dict.get(self, idx, self._default)


class IdFeatures:
    """
    We will change the id of item and shop so that it becomes easier for trees to split data.
    """
    ABSENT_ITEM_ID_VALUE = -1000

    def __init__(self, sales_df: pd.DataFrame, items_df: pd.DataFrame, num_clusters: Tuple[int, int] = (16, 16)):

        self._sales_df = sales_df
        self._items_df = items_df
        self._num_clusters = num_clusters

        # given a category, list of items
        self._category_to_item_df = items_df.groupby('item_category_id')['item_id'].apply(list)

        # given an item, a category.
        self._item_to_category_df = items_df.set_index('item_id')['item_category_id']

        self._item_id_old_to_new = None
        self._shop_id_old_to_new = None
        self._item_id_alternate = DefaultDict()
        self._item_id_alternate_dist = DefaultDict()

        # first time occuring features
        self._item_fm_df = None
        self._item_shop_fm_df = None

        self.fit()

    def _fit_first_time_occuring_features(self):
        assert 'orig_item_id' in self._sales_df

        temp_df = self._sales_df[self._sales_df.item_cnt_day > 0][['orig_item_id', 'shop_id', 'date_block_num']]

        self._item_fm_df = temp_df.groupby(['orig_item_id'])['date_block_num'].min().to_frame('fm').reset_index()
        self._item_shop_fm_df = temp_df.groupby(['orig_item_id',
                                                 'shop_id'])['date_block_num'].min().to_frame('fm').reset_index()
        assert 'orig_item_id' in self._item_fm_df
        assert 'orig_item_id' in self._item_shop_fm_df

    def _fit(self, id_name):
        monthly_df = self._sales_df.groupby(['date_block_num', id_name])['item_cnt_day'].sum().unstack().fillna(0)
        non_zero_months = (monthly_df > 0).sum()

        avg_monthly_sales_df = (monthly_df.sum() / non_zero_months).to_frame('avg_sales')
        avg_monthly_sales_df = avg_monthly_sales_df.sort_values('avg_sales')
        avg_monthly_sales_df['new_id'] = list(range(avg_monthly_sales_df.shape[0]))
        return DefaultDict(avg_monthly_sales_df['new_id'].to_dict())

    def find_missing_ids(self, test_sales_df):
        missing_ids = list(set(test_sales_df['item_id'].unique()) - set(self._sales_df['item_id'].unique()))
        missing_ids.sort()
        return missing_ids

    def set_alternate_ids(self, item_names_en: List[str], test_sales_df: pd.DataFrame):
        """
        Some item_ids were not present in training data. We need to find substitute ids.
        """
        # default self dict
        self._item_id_alternate = dict(pd.concat([self._items_df['item_id'], self._items_df['item_id']], axis=1).values)
        self._item_id_alternate_dist = {}

        # These ids don't appear in train data.
        extra_item_ids = self.find_missing_ids(test_sales_df)

        def get_id_distance(unavailable_id):
            distance = []
            ids = []

            target_str = item_names_en[unavailable_id].lower()
            available_neighbour_ids = self._category_to_item_df[self._item_to_category_df[unavailable_id]]
            for i_id in available_neighbour_ids:
                if i_id in extra_item_ids:
                    continue
                ids.append(i_id)
                distance.append(lev.distance(item_names_en[i_id].lower(), target_str))

            min_id_index = np.argmin(distance)
            min_id = ids[min_id_index]

            return (unavailable_id, min_id, distance[min_id_index])

        for unavailable_id in extra_item_ids:
            ans = get_id_distance(unavailable_id)
            self._item_id_alternate[ans[0]] = ans[1]
            self._item_id_alternate_dist[ans[0]] = ans[2]

    def fit(self):
        self._item_id_old_to_new = self._fit('item_id')
        self._shop_id_old_to_new = self._fit('shop_id')

    def get_fm_features(self, df, item_id_and_shop_id=False):
        """
        Adds  first month features to df.
        df must have ['orig_item_id','shop_id','date_block_num'] columns
        """
        assert 'orig_item_id' in df
        if self._item_fm_df is None:
            self._fit_first_time_occuring_features()

        merge_df = self._item_shop_fm_df if item_id_and_shop_id else self._item_fm_df
        on_columns = ['orig_item_id', 'shop_id'] if item_id_and_shop_id else ['orig_item_id']
        f_nm_prefix = '_'.join(on_columns) + '_'
        df = pd.merge(df.reset_index(), merge_df, on=on_columns, how='left').set_index('index')

        old_col = f_nm_prefix + 'oldness'
        fm_col = f_nm_prefix + 'is_fm'

        df[old_col] = df['date_block_num'] - df['fm']
        # We will set oldness to 0 for which we don't have the data.
        df[old_col] = df[old_col].fillna(0).astype(int)

        df[fm_col] = df[old_col] == 0

        return df.drop('fm', axis=1)

    def transform_item_id_to_alternate_id_dict(self):
        # some item ids donot exist in the training data. For them, we need to map them to appropriate items
        return self._item_id_alternate

    def transform_item_id_dict(self):
        return self._item_id_old_to_new

    def transform_shop_id_dict(self):
        return self._shop_id_old_to_new
