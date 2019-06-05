from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import nn_features
from nn_features import NNModels, NNModel, add_item_dbn_id, add_item_shop_dbn_id, get_neighbor_item_ids, set_nn_feature


def dummy_items():
    items_text = [
        '"1C-Bitrix Site Manager - Small Business [PC, Digital Version]"',
        '"1C-Bitrix Site Manager - Standard [PC, Digital Version]"',
        '"1C-Bitrix Site Manager - Start (Bitrix) [PC, Digital Version]"',
        'License 1C-Bitrix Site Manager - Small Business',
        'License 1C-Bitrix Site Manager - Standard',
        'License 1C-Bitrix Site Manager - Start',
        'License 1C-Bitrix Site Manager - Expert',
        '"FIFA Manager 10 [PC, Digital Version]"',
        '"FIFA Manager 13 [PC, Digital Version]"',
        '"FIFA Manager 14 [PC, Digital Version]"',
    ]
    items_df = pd.DataFrame([], index=list(range(len(items_text))))
    items_df['item_name'] = items_text
    items_df['item_id'] = items_df.index.tolist()
    items_df['item_category_id'] = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    return items_df


def get_items_and_features():
    items_df = dummy_items()
    index_list = items_df.index.tolist()
    np.random.shuffle(index_list)
    items_df = items_df.loc[index_list]

    tf = TfidfVectorizer()
    item_features = tf.fit_transform(items_df.item_name.values)
    return items_df, item_features


def test_NNModel():
    items_df, item_features = get_items_and_features()
    model = NNModel(item_features, items_df.item_id.values, 3, 'minkowski')
    item_ids = items_df.item_id.tolist()
    i = item_ids.index(7)
    neighbour_ids, _ = model.predict(item_features[i:i + 1])
    assert set([7, 8, 9]) == set(neighbour_ids)


def test_NNModels_should_create_one_month_lagged_models_of_one_category():
    nn_features.CLUSTER_MONTH_WINDOW = 6
    columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        [11, 0, 0, 1.0],
        [11, 0, 1, 2.0],
        [11, 0, 3, 1.0],
        [11, 0, 4, 1.0],
        [11, 0, 5, 5.0],
        [11, 0, 8, 1.0],
        [11, 0, 9, 2.0],

        # 2nd month
        [12, 0, 2, 1.0],
        [12, 0, 6, 2.0],
        [12, 0, 7, 1.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    items_df, item_features = get_items_and_features()
    models = NNModels(item_features, sales_df, items_df, 3)
    neighbors = models.neighbors
    for dbn in range(35):
        if dbn not in [12, 13, 14, 15, 16, 17, 18]:
            assert neighbors[dbn][0] is None
            assert neighbors[dbn][1] is None
            assert neighbors[dbn][2] is None
        else:
            assert neighbors[dbn][0] is not None
            assert neighbors[dbn][1] is not None
            assert neighbors[dbn][2] is not None

        if dbn == 12:
            assert set(neighbors[dbn][0]._item_ids) == set([0, 1])
            assert set(neighbors[dbn][1]._item_ids) == set([3, 4, 5])
            assert set(neighbors[dbn][2]._item_ids) == set([8, 9])
        if dbn in [13, 14, 15, 16, 17]:
            assert set(neighbors[dbn][0]._item_ids) == set([0, 1, 2])
            assert set(neighbors[dbn][1]._item_ids) == set([3, 4, 5, 6])
            assert set(neighbors[dbn][2]._item_ids) == set([8, 9, 7])
        if dbn == 18:
            assert set(neighbors[dbn][0]._item_ids) == set([2])
            assert set(neighbors[dbn][1]._item_ids) == set([6])
            assert set(neighbors[dbn][2]._item_ids) == set([7])


def test_new_id_computation():
    df = pd.DataFrame([[955, 25, 12]], columns=['item_id', 'shop_id', 'date_block_num'])

    add_item_shop_dbn_id(df)
    assert df.iloc[0]['item_shop_dbn_idx'] == 955 * 100 * 35 + 25 * 35 + 12

    add_item_dbn_id(df)
    assert df.iloc[0]['item_dbn_idx'] == 955 * 35 + 12


def test_get_neighbor_item_ids():
    nn_features.CLUSTER_MONTH_WINDOW = 6
    columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        [11, 0, 0, 1.0],
        [11, 0, 1, 2.0],
        [11, 0, 3, 1.0],
        [11, 0, 4, 1.0],
        [11, 0, 5, 5.0],
        [11, 0, 8, 1.0],
        [11, 0, 9, 2.0],

        # 2nd month
        [12, 0, 2, 1.0],
        [12, 0, 6, 2.0],
        [12, 0, 7, 1.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    items_df, items_text_X = get_items_and_features()
    models = NNModels(items_text_X, sales_df, items_df, 4)

    item_to_category_id = items_df.set_index('item_id')['item_category_id'].to_dict()
    assert {1} == set(get_neighbor_item_ids(12, 0, models.neighbors, items_text_X, item_to_category_id))
    assert {0} == set(get_neighbor_item_ids(12, 1, models.neighbors, items_text_X, item_to_category_id))
    assert {0, 1} == set(get_neighbor_item_ids(12, 2, models.neighbors, items_text_X, item_to_category_id))

    assert {4, 5} == set(get_neighbor_item_ids(12, 3, models.neighbors, items_text_X, item_to_category_id))
    assert {3, 5} == set(get_neighbor_item_ids(12, 4, models.neighbors, items_text_X, item_to_category_id))
    assert {3, 4} == set(get_neighbor_item_ids(12, 5, models.neighbors, items_text_X, item_to_category_id))
    assert {3, 4, 5} == set(get_neighbor_item_ids(12, 6, models.neighbors, items_text_X, item_to_category_id))

    assert {8, 9} == set(get_neighbor_item_ids(12, 7, models.neighbors, items_text_X, item_to_category_id))
    assert {9} == set(get_neighbor_item_ids(12, 8, models.neighbors, items_text_X, item_to_category_id))
    assert {8} == set(get_neighbor_item_ids(12, 9, models.neighbors, items_text_X, item_to_category_id))
    for dbn in range(13, 18):
        assert {1, 2} == set(get_neighbor_item_ids(dbn, 0, models.neighbors, items_text_X, item_to_category_id))
        assert {0, 2} == set(get_neighbor_item_ids(dbn, 1, models.neighbors, items_text_X, item_to_category_id))
        assert {0, 1} == set(get_neighbor_item_ids(dbn, 2, models.neighbors, items_text_X, item_to_category_id))

        assert {4, 5, 6} == set(get_neighbor_item_ids(dbn, 3, models.neighbors, items_text_X, item_to_category_id))
        assert {3, 5, 6} == set(get_neighbor_item_ids(dbn, 4, models.neighbors, items_text_X, item_to_category_id))
        assert {3, 4, 6} == set(get_neighbor_item_ids(dbn, 5, models.neighbors, items_text_X, item_to_category_id))
        assert {3, 4, 5} == set(get_neighbor_item_ids(dbn, 6, models.neighbors, items_text_X, item_to_category_id))

        assert {8, 9} == set(get_neighbor_item_ids(dbn, 7, models.neighbors, items_text_X, item_to_category_id))
        assert {7, 9} == set(get_neighbor_item_ids(dbn, 8, models.neighbors, items_text_X, item_to_category_id))
        assert {7, 8} == set(get_neighbor_item_ids(dbn, 9, models.neighbors, items_text_X, item_to_category_id))


def test_set_nn_feature():
    feature_col = 'item_cnt_day'
    n_neighbors = [4, 1]

    columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        [11, 0, 0, 1.0],
        [11, 0, 1, 2.0],
        [11, 1, 0, 5.0],
        # 2nd category
        [11, 0, 3, 3.0],
        [11, 0, 4, 1.0],
        [11, 1, 5, 5.0],
        # 3rd category
        [11, 0, 8, 1.0],
        [11, 0, 9, 2.0],

        # 2nd month
        [12, 0, 2, 1.0],
        [12, 0, 6, 2.0],
        [12, 0, 7, 1.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    items_df, items_text_data = get_items_and_features()
    sales_df = pd.merge(
        sales_df.reset_index(), items_df[['item_id', 'item_category_id']], how='left', on='item_id').set_index('index')

    sales_df[['item_id', 'shop_id', 'date_block_num',
              'item_category_id']] = sales_df[['item_id', 'shop_id', 'date_block_num', 'item_category_id']].astype(
                  np.int32)

    X_df = sales_df.copy()
    X_df['date_block_num'] += 1
    X_df['orig_item_id_is_fm'] = False

    sales_df.drop('item_category_id', axis=1, inplace=True)

    set_nn_feature(X_df, feature_col, items_text_data, sales_df, items_df, n_neighbors)

    new_col1 = '{}Neighbor_{}'.format(4, feature_col)
    assert X_df[X_df.date_block_num == 11].empty
    assert X_df.iloc[0][new_col1] == 2 / 1
    assert X_df.iloc[1][new_col1] == 1 / 1
    assert X_df.iloc[2][new_col1] == 2 / 1
    assert X_df.iloc[3][new_col1] == (5 + 1) / 2
    assert X_df.iloc[4][new_col1] == (5 + 3) / 2
    assert X_df.iloc[5][new_col1] == (1 + 3) / 2
    assert X_df.iloc[6][new_col1] == 2 / 1
    assert X_df.iloc[7][new_col1] == 1 / 1
    assert X_df.iloc[8][new_col1] == -10
    assert X_df.iloc[9][new_col1] == -10

    new_col2 = '{}Neighbor_{}'.format(1, feature_col)
    assert X_df[X_df.date_block_num == 11].empty
    assert X_df.iloc[0][new_col2] == 2 / 1
    assert X_df.iloc[1][new_col2] == 1 / 1
    assert X_df.iloc[2][new_col2] == 2 / 1
    assert X_df.iloc[3][new_col2] == 5 / 1 or X_df.iloc[3][new_col2] == 1 / 1
    assert X_df.iloc[4][new_col2] == 3 / 1 or X_df.iloc[4][new_col2] == 5 / 1
    assert X_df.iloc[5][new_col2] == 3 / 1 or X_df.iloc[5][new_col2] == 1 / 1
    assert X_df.iloc[6][new_col2] == 2 / 1
    assert X_df.iloc[7][new_col2] == 1 / 1
    assert X_df.iloc[8][new_col2] == -10
    assert X_df.iloc[9][new_col2] == -10
