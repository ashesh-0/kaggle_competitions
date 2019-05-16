import pandas as pd
from mean_encoding import MeanEncoding


def dummy_data():
    columns = ['shop_id', 'item_id', 'item_category_id', 'target']

    # Train
    train_data = [
        [0, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 2, 1, 3],
        [1, 0, 0, 4],
        [1, 1, 0, 5],
        [2, 0, 0, 6],
        [3, 2, 1, 7],
        [3, 2, 1, 3],
    ]

    # Val
    val_data = [
        [0, 0, 0, 3],
        [0, 1, 0, 3],
        [0, 2, 1, 3],
        [1, 0, 0, 3],
        [1, 1, 0, 3],
        [2, 0, 0, 3],
        [3, 2, 1, 3],
    ]

    train_df = pd.DataFrame(train_data, columns=columns)
    val_df = pd.DataFrame(val_data, columns=columns)

    return train_df, val_df


def test_mean_encoding_should_fit_global_correctly():
    MeanEncoding.MAX_ITEM_CATEGORY_ID = 5
    MeanEncoding.MAX_ITEM_ID = 5
    MeanEncoding.MAX_SHOP_ID = 5

    print('')
    train_df, val_df = dummy_data()

    train_y = train_df['target']
    val_y = val_df['target']

    train_df.drop('target', axis=1, inplace=True)
    val_df.drop('target', axis=1, inplace=True)

    train_c = ['shop_id', 'item_id', 'item_category_id']
    me = MeanEncoding(train_df[train_c], train_y, val_df[train_c], val_y)

    assert me._item_category_encoding_map[0] == 18 / 5
    assert me._item_category_encoding_map[1] == 13 / 3

    assert me._item_encoding_map[0] == 11 / 3
    assert me._item_encoding_map[1] == 7 / 2
    assert me._item_encoding_map[2] == 13 / 3

    assert me._shop_encoding_map[0] == 6 / 3
    assert me._shop_encoding_map[1] == 9 / 2
    assert me._shop_encoding_map[2] == 6
    assert me._shop_encoding_map[3] == 5

    assert me._item_shop_encoding_map[0] == 1
    assert me._item_shop_encoding_map[1] == 4
    assert me._item_shop_encoding_map[2] == 6
    assert me._item_shop_encoding_map[100] == 2
    assert me._item_shop_encoding_map[101] == 5
    assert me._item_shop_encoding_map[200] == 3
    assert me._item_shop_encoding_map[203] == 5

    assert me._shop_category_encoding_map[0] == 1.5
    assert me._shop_category_encoding_map[1] == 3

    assert me._shop_category_encoding_map[100] == 4.5
    assert me._shop_category_encoding_map[101] == 0

    assert me._shop_category_encoding_map[200] == 6
    assert me._shop_category_encoding_map[201] == 0

    assert me._shop_category_encoding_map[300] == 0
    assert me._shop_category_encoding_map[301] == 5

    print(train_df)


def test_get_train_data_should_work():
    MeanEncoding.MAX_ITEM_CATEGORY_ID = 5
    MeanEncoding.MAX_ITEM_ID = 5
    MeanEncoding.MAX_SHOP_ID = 5

    train_df, val_df = dummy_data()

    train_y = train_df['target']
    val_y = val_df['target']

    train_df.drop('target', axis=1, inplace=True)
    val_df.drop('target', axis=1, inplace=True)

    train_c = ['shop_id', 'item_id', 'item_category_id']
    me = MeanEncoding(train_df[train_c], train_y, val_df[train_c], val_y)
    X, y = me.get_train_data()
    print(X)
    assert X.index.equals(y.index)


def test_get_val_data_should_work():
    MeanEncoding.MAX_ITEM_CATEGORY_ID = 5
    MeanEncoding.MAX_ITEM_ID = 5
    MeanEncoding.MAX_SHOP_ID = 5

    train_df, val_df = dummy_data()

    train_y = train_df['target']
    val_y = val_df['target']

    train_df.drop('target', axis=1, inplace=True)
    val_df.drop('target', axis=1, inplace=True)

    train_c = ['shop_id', 'item_id', 'item_category_id']
    me = MeanEncoding(train_df[train_c], train_y, val_df[train_c], val_y)
    val_X_df, val_y_df = me.get_val_data()

    val_X_df['item_shop_id'] = 100 * val_X_df['item_id'] + val_X_df['shop_id']
    item_encoding = val_X_df['item_id'].map({0: 11 / 3, 1: 7 / 2, 2: 13 / 3})
    shop_encoding = val_X_df['shop_id'].map({0: 2, 1: 4.5, 2: 6, 3: 5})
    categ_encoding = val_X_df['item_category_id'].map({0: 18 / 5, 1: 13 / 3})
    item_shop_encoding = val_X_df['item_shop_id'].map({0: 1, 1: 4, 2: 6, 100: 2, 101: 5, 200: 3, 203: 5})

    assert val_X_df['item_id_mean_enc'].equals(item_encoding)
    assert val_X_df['shop_id_mean_enc'].equals(shop_encoding)
    assert val_X_df['item_category_id_mean_enc'].equals(categ_encoding)
    assert val_X_df['item_shop_id_mean_enc'].equals(item_shop_encoding)
