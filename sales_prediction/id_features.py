from sklearn.cluster.bicluster import SpectralBiclustering
import Levenshtein as lev
from typing import List, Tuple
import numpy as np
import pandas as pd


class IdFeatures:
    """
    We will change the id of item and shop so that it becomes easier for trees to split data.
    """

    def __init__(self, sales_df: pd.DataFrame, items_df: pd.DataFrame, num_clusters: Tuple[int, int] = (4, 4)):

        self._sales_df = sales_df
        self._items_df = items_df
        self._num_clusters = num_clusters

        # given a category, list of items
        self._category_to_item_df = items_df.groupby('item_category_id')['item_id'].apply(list)

        # given an item, a category.
        self._item_to_category_df = items_df.set_index('item_id')['item_category_id']

        self._item_id_old_to_new = None
        self._shop_id_old_to_new = None
        self._item_id_alternate = {}
        self._item_id_alternate_dist = {}

        # Using biclustering, we will assign cluster_ids which can then be used.
        self._category_id_to_cluster = {}
        self._shop_id_to_cluster = {}

        self.fit()

    def _fit_cluster(self):
        if 'item_category_id' not in self._sales_df:
            merged_df = pd.merge(self._sales_df, self._items_df, how='left', on='item_id')
        else:
            merged_df = self._sales_df

        df = merged_df.groupby(['shop_id', 'item_category_id', 'month'])['item_cnt_day'].sum().reset_index()
        df = df.groupby(['shop_id', 'item_category_id'])['item_cnt_day'].mean()
        df[df > 20] = 20

        data_df = df.unstack()
        data = data_df.fillna(data_df.mean()).values

        model = SpectralBiclustering(n_clusters=self._num_clusters, method='log', random_state=0)
        model.fit(data)

        self._shop_id_to_cluster = dict(list(zip(data_df.index.tolist(), model.row_labels_)))
        self._category_id_to_cluster = dict(list(zip(data_df.columns.tolist(), model.column_labels_)))

    def _fit(self, id_name):
        monthly_df = self._sales_df.groupby(['date_block_num', id_name])['item_cnt_day'].sum().unstack().fillna(0)
        non_zero_months = (monthly_df > 0).sum()

        avg_monthly_sales_df = (monthly_df.sum() / non_zero_months).to_frame('avg_sales')
        avg_monthly_sales_df = avg_monthly_sales_df.sort_values('avg_sales')
        avg_monthly_sales_df['new_id'] = list(range(avg_monthly_sales_df.shape[0]))
        return avg_monthly_sales_df['new_id'].to_dict()

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

        def update_progress(ans):
            self._item_id_alternate[ans[0]] = ans[1]
            self._item_id_alternate_dist[ans[0]] = ans[2]

        def get_id_distance(unavailable_id):
            distance = []
            target_str = item_names_en[unavailable_id].lower()
            available_neighbour_ids = self._category_to_item_df[self._item_to_category_df[unavailable_id]]
            for i_id in available_neighbour_ids:
                if i_id in extra_item_ids:
                    continue
                distance.append(lev.distance(item_names_en[i_id].lower(), target_str))
            min_id = np.argmin(distance)
            return (unavailable_id, min_id, distance[min_id])

        for unavailable_id in extra_item_ids:
            update_progress(get_id_distance(unavailable_id))

    def fit(self):
        self._item_id_old_to_new = self._fit('item_id')
        self._shop_id_old_to_new = self._fit('shop_id')

        self._fit_cluster()

    def transform_item_id_to_alternate_id_dict(self):
        # some item ids donot exist in the training data. For them, we need to map them to appropriate items
        return self._item_id_alternate

    def transform_category_id_to_cluster_dict(self):
        return self._category_id_to_cluster

    def transform_item_id_dict(self):
        return self._item_id_old_to_new

    def transform_shop_id_dict(self):
        return self._shop_id_old_to_new

    def transform_shop_id_to_cluster_dict(self):
        return self._shop_id_to_cluster


# if __name__ == '__main__':
#     cols = ['date_block_num', 'item_id', 'shop_id', 'item_cnt_day']
#     df = pd.DataFrame(
#         [
#             [0, 0, 0, 5],
#             [0, 0, 0, 10],
#             [1, 0, 0, 4],
#             [1, 0, 0, 6],
#             [0, 1, 0, 1],
#             [0, 1, 0, 2],
#             [1, 1, 0, 4],
#             [1, 1, 0, 6],
#             [0, 2, 1, 10],
#             [0, 2, 1, 2],
#             [1, 2, 1, 4],
#             [1, 2, 1, 6],
#         ],
#         columns=cols)

#     f = IdFeatures(df)
#     print(f.get_features(0, 0))
