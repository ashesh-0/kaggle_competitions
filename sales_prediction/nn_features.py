"""
Nearest neighbour features. It should work well for those item_id's which are not present
in test data. This corresponds to new item_ids.
I've month, item_id, shop_id, item_category_id. So we can use features
only based on this.
"""
from typing import List, Tuple
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
# from multiprocessing import Pool
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# import cProfile, pstats

DEFAULT_VALUE = -10
CLUSTER_MONTH_WINDOW = 1


def tokenizer(s):
    tokens = re.split(' |-|/|:', s)
    tokens = [x.strip('[(*"./)]') for x in tokens]
    while '' in tokens:
        tokens.remove('')
    for word in ['and', 'for', 'of', 'the', '.', ',', '+']:
        while word in tokens:
            tokens.remove(word)

    return tokens


def get_existing_items(sales_df, date_block_num):
    """
    Look at past 6 months excluding this date_block_num
    """
    dbns = list(range(date_block_num - CLUSTER_MONTH_WINDOW, date_block_num))
    return set(sales_df[sales_df.date_block_num.isin(dbns)].item_id.unique())


class NNModel:
    def __init__(self, item_ids_text_features, item_ids, n_neighbors, metric):
        self._item_ids = np.array(item_ids)
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=1, algorithm='auto')
        self.model.fit(item_ids_text_features)

    def predict(self, item_id_text_feature) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Returns:
            neigh_ids: array of neighbor item_ids
        """
        NN_output = self.model.kneighbors(item_id_text_feature)
        neighs = NN_output[1]
        # getting item_ids
        neigh_ids = [list(self._item_ids[id_set]) for id_set in neighs]

        neighs_dist = NN_output[0]
        return (neigh_ids, neighs_dist)


class NNModels:
    """
    Create NN models for each category, date_block_num
    """

    def __init__(self, items_text_data, sales_df, items_df, n_neighbors):
        # for each category, learn a nearest neighbour on a rolling basis.
        self._metric = 'minkowski'
        self._items_text = items_text_data
        self._sales = sales_df
        self._items = items_df
        self._n_neighbors = n_neighbors
        self._sales = pd.merge(self._sales.reset_index(), self._items, on='item_id', how='left').set_index('index')
        self._category_to_item_ids = self._sales.groupby('item_category_id')['item_id'].apply(set).to_dict()
        self.neighbors = {}

        self.run()

    def run(self):
        for dbn in range(35):
            self.neighbors[dbn] = {}
            item_ids = get_existing_items(self._sales, dbn)
            for item_category_id in self._category_to_item_ids:
                self.neighbors[dbn][item_category_id] = None
                cat_item_ids = list(item_ids.intersection(self._category_to_item_ids[item_category_id]))

                if len(cat_item_ids) == 0:
                    continue

                X = self._items_text[cat_item_ids]
                model = NNModel(X, cat_item_ids, min(self._n_neighbors, X.shape[0]), self._metric)
                self.neighbors[dbn][item_category_id] = model

    # map : item_id_date_block_num => [item_ids]
    # Pick the latest value


def get_item_shop_dbn_id(item_id, shop_id, date_block_num):
    return item_id * 100 * 35 + shop_id * 35 + date_block_num


def get_item_dbn_id(item_id, date_block_num):
    return item_id * 35 + date_block_num


def add_item_shop_dbn_id(df):
    df['item_shop_dbn_idx'] = get_item_shop_dbn_id(df['item_id'], df['shop_id'], df['date_block_num'])


def add_item_dbn_id(df):
    df['item_dbn_idx'] = get_item_dbn_id(df['item_id'], df['date_block_num'])


def get_nn_features_data(X_df, feature_col):
    item_features = X_df.groupby(['item_id', 'date_block_num'])[[feature_col]].mean().reset_index()
    shop_item_features = X_df[['item_id', 'shop_id', 'date_block_num', feature_col]].copy()

    # create index
    add_item_shop_dbn_id(shop_item_features)
    shop_item_features.set_index('item_shop_dbn_idx', inplace=True)

    add_item_dbn_id(item_features)
    item_features.set_index('item_dbn_idx', inplace=True)
    return (shop_item_features[feature_col].to_dict(), item_features[feature_col].to_dict())


def get_neighbor_item_ids(
        date_block_num,
        item_category_id,
        item_ids,
        neighbors,
        items_text_X,
):
    model = neighbors[date_block_num][item_category_id]
    if model is None:
        return [[]] * len(item_ids)

    text_features = items_text_X[item_ids]

    neighbor_item_ids, _ = model.predict(text_features)
    for idx, item_id in enumerate(item_ids):
        if item_id in neighbor_item_ids[idx]:
            neighbor_item_ids[idx].remove(item_id)
            # TODO: remove from distance.

    return neighbor_item_ids


def _get_one_row_feature(date_block_num, shop_id, shop_item_features, item_features, neighbor_item_ids):
    feature = np.zeros(len(neighbor_item_ids))
    for i, n_item_id in enumerate(neighbor_item_ids):
        item_shop_dbn_id = get_item_shop_dbn_id(n_item_id, shop_id, date_block_num)
        if item_shop_dbn_id in shop_item_features:
            feature[i] = shop_item_features[item_shop_dbn_id]
        else:
            item_dbn_id = get_item_dbn_id(n_item_id, date_block_num)
            assert item_dbn_id in item_features, '{}-{} item_dbn_id not present '.format(n_item_id, date_block_num)
            feature[i] = item_features[item_dbn_id]

    return np.mean(feature)


def get_one_dbn_category_feature(
        date_block_num,
        item_category_id,
        item_ids,
        shop_ids,
        shop_item_features,
        item_features,
        neighbors,
        items_text_X,
):
    output = np.ones(len(shop_ids)) * DEFAULT_VALUE
    neighbor_item_ids = get_neighbor_item_ids(date_block_num, item_category_id, item_ids, neighbors, items_text_X)
    for i, shop_id_neighbors in enumerate(zip(shop_ids, neighbor_item_ids)):
        shop_id, neighbors = shop_id_neighbors
        if len(neighbors) == 0:
            continue
        output[i] = _get_one_row_feature(date_block_num, shop_id, shop_item_features, item_features, neighbors)
    return output


def set_nn_feature(
        X_df,
        feature_col,
        items_text_data,
        sales_df,
        items_df,
        n_neighbors,
        n_jobs=1,
):
    # create models for getting neighbor item_ids
    models = NNModels(items_text_data, sales_df, items_df, n_neighbors)
    print('Models created')

    # get features for item ids.
    (shop_item_features, item_features) = get_nn_features_data(X_df, feature_col)
    print('Features computed')

    feature = np.zeros(X_df.shape[0])
    index = np.zeros(X_df.shape[0])
    f_idx = 0
    # pr = cProfile.Profile()
    # pr.enable()
    for dbn in tqdm(range(35)):
        dbn_X = X_df[X_df.date_block_num == dbn]
        for item_category_id in range(84):
            temp_X = dbn_X[dbn_X.item_category_id == item_category_id]
            if temp_X.empty:
                continue

            one_block_data = get_one_dbn_category_feature(
                dbn,
                item_category_id,
                temp_X.item_id.values,
                temp_X.shop_id.values,
                shop_item_features,
                item_features,
                models.neighbors,
                items_text_data,
            )
            feature[f_idx:f_idx + one_block_data.shape[0]] = one_block_data
            index[f_idx:f_idx + one_block_data.shape[0]] = temp_X.index.tolist()
            f_idx += one_block_data.shape[0]

    # pr.disable()
    # sortby = 'cumulative'
    # with open('log.txt', 'w') as stream:
    #     ps = pstats.Stats(pr, stream=stream).sort_stats(sortby)
    #     ps.print_stats()

    X_df['neighbor_' + feature_col] = pd.Series(feature, index=index)


if __name__ == '__main__':
    from constants import SALES_FPATH, ITEMS_FPATH, COMPETITION_DATA_DIRECTORY
    items_df = pd.read_csv(ITEMS_FPATH)
    sales_df = pd.read_csv(SALES_FPATH)
    sales_df = sales_df.groupby(['item_id', 'shop_id', 'date_block_num']).last().reset_index()
    X_df = sales_df.copy()
    X_df = pd.merge(X_df, items_df[['item_id', 'item_category_id']], how='left', on='item_id')

    item_names = open(COMPETITION_DATA_DIRECTORY + '/item_name.ru.en.txt', 'r').read().splitlines()

    tf = TfidfVectorizer(tokenizer=tokenizer, min_df=0.0003)
    items_text_data = tf.fit_transform(item_names)

    n_neighbors = 5
    feature_col = 'item_cnt_day'
    X_minimal_df = X_df[['date_block_num', 'item_id', 'shop_id', 'item_category_id', feature_col]].copy()
    X_minimal_df[['date_block_num', 'item_id', 'shop_id']] = X_minimal_df[['date_block_num', 'item_id',
                                                                           'shop_id']].astype(np.int32)
    set_nn_feature(X_minimal_df, feature_col, items_text_data, sales_df, items_df, n_neighbors)
