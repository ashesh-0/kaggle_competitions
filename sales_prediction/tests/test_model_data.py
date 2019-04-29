import pandas as pd
from model_data import get_X_y
from constant_lists import CATEGORIES_EN, SHOPS_EN


def get_dummy_items():
    return pd.DataFrame(
        [
            [0, 0],
            [1, 0],
            [2, 1],
        ], columns=['item_id', 'item_category_id'])


def get_dummy_sales():
    # 2 shops, 0,1
    # 3 products
    # 5 days.
    columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
    data = [
        ['02.01.2013', 0, 0, 0, 10, 1.0],
        ['02.01.2013', 0, 0, 1, 100, 1.0],
        ['02.01.2013', 0, 0, 2, 1000, 1.0],
        ['02.01.2013', 0, 1, 0, 10, 1.0],
        ['02.01.2013', 0, 1, 1, 100, 1.0],
        ['02.01.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['03.01.2013', 0, 0, 0, 20, 2.0],
        ['03.01.2013', 0, 0, 1, 100, 1.0],
        ['03.01.2013', 0, 0, 2, 1000, 1.0],
        ['03.01.2013', 0, 1, 0, 20, 5.0],
        ['03.01.2013', 0, 1, 1, 100, 2.0],
        ['03.01.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['04.01.2013', 0, 0, 0, 20, 2.0],
        ['04.01.2013', 0, 0, 1, 120, 1.0],
        ['04.01.2013', 0, 0, 2, 1000, 1.0],
        ['04.01.2013', 0, 1, 0, 20, 5.0],
        ['04.01.2013', 0, 1, 1, 120, 2.0],
        ['04.01.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['05.01.2013', 0, 0, 0, 20, 2.0],
        ['05.01.2013', 0, 0, 1, 100, 1.0],
        ['05.01.2013', 0, 0, 2, 1000, 1.0],
        ['05.01.2013', 0, 1, 0, 20, 5.0],
        ['05.01.2013', 0, 1, 1, 100, 2.0],
        ['05.01.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['06.01.2013', 0, 0, 0, 20, 2.0],
        ['06.01.2013', 0, 0, 1, 100, 1.0],
        ['06.01.2013', 0, 0, 2, 1500, 1.0],
        ['06.01.2013', 0, 1, 0, 20, 5.0],
        ['06.01.2013', 0, 1, 1, 100, 2.0],
        ['06.01.2013', 0, 1, 2, 1500, 10],
    ]
    return pd.DataFrame(data, columns=columns)


def test_get_X_y():
    get_X_y(get_dummy_sales(), get_dummy_items(), CATEGORIES_EN, SHOPS_EN)
