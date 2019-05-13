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
    print('')
    train_df, val_df = dummy_data()

    train_y = train_df['target']
    val_y = val_df['target']

    train_df.drop('target', axis=1, inplace=True)
    val_df.drop('target', axis=1, inplace=True)

    train_c = ['shop_id', 'item_id', 'item_category_id']
    me = MeanEncoding(train_df[train_c], train_y, val_df[train_c], val_y)

    assert me._item_category_encoding_map[0] == 18 / 5
    assert me._item_category_encoding_map[1] == 10 / 2

    assert me._item_encoding_map[0] == 11 / 3
    assert me._item_encoding_map[1] == 7 / 2
    assert me._item_encoding_map[2] == 10 / 2

    assert me._shop_encoding_map[0] == 6 / 3
    assert me._shop_encoding_map[1] == 9 / 2
    assert me._shop_encoding_map[2] == 6
    assert me._shop_encoding_map[3] == 7

    print(train_df)


def test_mean_should_get_val_correctly():

    train_df, val_df = dummy_data()

    train_y = train_df['target']
    val_y = val_df['target']

    train_df.drop('target', axis=1, inplace=True)
    val_df.drop('target', axis=1, inplace=True)

    train_c = ['shop_id', 'item_id', 'item_category_id']
    me = MeanEncoding(train_df[train_c], train_y, val_df[train_c], val_y)
    val_X_df, val_y_df = me.get_val_data()

    item_encoding = val_X_df['item_id'].map({0: 11 / 3, 1: 7 / 2, 2: 10 / 2})
    shop_encoding = val_X_df['shop_id'].map({0: 2, 1: 4.5, 2: 6, 3: 7})
    categ_encoding = val_X_df['item_category_id'].map({0: 18 / 5, 1: 5})

    assert val_X_df['item_id_mean_enc'].equals(item_encoding)
    assert val_X_df['shop_id_mean_enc'].equals(shop_encoding)
    assert val_X_df['item_category_id_mean_enc'].equals(categ_encoding)
