import pandas as pd

from model_data import ModelData
from constant_lists import CATEGORIES_EN, SHOPS_EN
from .test_utils import get_dummy_sales, get_dummy_items, get_dummy_category, get_dummy_shops
import os
from unittest.mock import patch


def compute_sequentially(args, **kwargs):
    def run(fn_args):
        fn, args, name = fn_args
        print('Starting', args)
        return fn(*args)

    output = []
    for arg in args:
        output.append(run(arg))
    return pd.concat(output, axis=1)


@patch('numeric_features.compute_concurrently', side_effect=compute_sequentially)
def test_get_X_y_runs(mock_fn1):
    sales_df = get_dummy_sales()
    items_df = get_dummy_items()
    category_en = get_dummy_category().item_category_name.values.tolist()
    shop_name_en = get_dummy_shops().shop_name.values.tolist()
    item_name_en = get_dummy_items().item_name.values.tolist()
    md = ModelData(sales_df, items_df, category_en, shop_name_en, item_name_en, num_clusters=(1, 1))

    X_df, y_df = md.get_train_X_y()
    print('X has shape', X_df.shape)
    print('Y has shape', y_df.shape)


# def test_get_X_test_should_change_item_id_correctly():
#     category_en = get_dummy_category().item_category_name.values.tolist()
#     item_name_en = get_dummy_items().item_name.values.tolist()

#     sales_train_df = get_dummy_sales()
#     _ = get_X_y(sales_train_df, get_dummy_items(), category_en, SHOPS_EN)

#     sales_test_df = pd.DataFrame(
#         [
#             [3, 0],
#             [3, 1],
#             [0, 1],
#         ], columns=['item_id', 'shop_id'])
#     test_df = get_X_test(sales_train_df, sales_test_df.copy(), get_dummy_items(), category_en, SHOPS_EN, item_name_en)

#     print(test_df)
#     test_df.sort_index(inplace=True)

#     assert test_df.shape[0] == sales_test_df.shape[0]
#     # Since 3 will map to 0. All of it should map to same item_id
#     assert len(test_df.item_id.unique()) == 1
#     assert sales_test_df['shop_id'].equals(test_df.shop_id)
