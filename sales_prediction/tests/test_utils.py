import pandas as pd


def get_dummy_items():
    return pd.DataFrame(
        [
            ['XYZ Donuts', 0, 0],
            ['ABC toothpaste', 1, 0],
            ['three idiots', 2, 1],
            ['XZ Donuts', 3, 0],
        ],
        columns=['item_name', 'item_id', 'item_category_id'])


def get_dummy_shops():
    return pd.DataFrame(
        [
            ['24/7 Welcome', 0],
            ['Pxmart', 1],
        ], columns=['shop_name', 'shop_id'])


def get_dummy_category():
    return pd.DataFrame(
        [
            ['Grocery', 0],
            ['Film', 1],
        ], columns=['item_category_name', 'item_category_id'])


def get_dummy_sales():
    # 2 shops, 0,1
    # 3 products
    # 2 months data, 5 days each.
    columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']
    data = [
        ['02.11.2013', 0, 0, 0, 10, 1.0],
        ['02.11.2013', 0, 0, 1, 100, 1.0],
        ['02.11.2013', 0, 0, 2, 1000, 1.0],
        ['02.11.2013', 0, 1, 0, 10, 1.0],
        ['02.11.2013', 0, 1, 1, 100, 1.0],
        ['02.11.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['03.11.2013', 0, 0, 0, 20, 2.0],
        ['03.11.2013', 0, 0, 1, 100, 1.0],
        ['03.11.2013', 0, 0, 2, 1000, 1.0],
        ['03.11.2013', 0, 1, 0, 20, 5.0],
        ['03.11.2013', 0, 1, 1, 100, 2.0],
        ['03.11.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['04.11.2013', 0, 0, 0, 20, 2.0],
        ['04.11.2013', 0, 0, 1, 120, 1.0],
        ['04.11.2013', 0, 0, 2, 1000, 1.0],
        ['04.11.2013', 0, 1, 0, 20, 5.0],
        ['04.11.2013', 0, 1, 1, 120, 2.0],
        ['04.11.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['05.11.2013', 0, 0, 0, 20, 2.0],
        ['05.11.2013', 0, 0, 1, 100, 1.0],
        ['05.11.2013', 0, 0, 2, 1000, 1.0],
        ['05.11.2013', 0, 1, 0, 20, 5.0],
        ['05.11.2013', 0, 1, 1, 100, 2.0],
        ['05.11.2013', 0, 1, 2, 1000, 1.0],
    ]

    data += [
        ['06.11.2013', 0, 0, 0, 20, 2.0],
        ['06.11.2013', 0, 0, 1, 100, 1.0],
        ['06.11.2013', 0, 0, 2, 1500, 1.0],
        ['06.11.2013', 0, 1, 0, 20, 5.0],
        ['06.11.2013', 0, 1, 1, 100, 2.0],
        ['06.11.2013', 0, 1, 2, 1500, 10],
    ]
    # month 2
    data += [
        ['02.10.2015', 1, 0, 0, 10, 1.0],
        ['02.10.2015', 1, 0, 1, 100, 2.0],
        ['02.10.2015', 1, 0, 2, 1000, 1.0],
        ['02.10.2015', 1, 1, 0, 10, 1.0],
        ['02.10.2015', 1, 1, 1, 100, 6.0],
        ['02.10.2015', 1, 1, 2, 1000, 1.0],
    ]

    data += [
        ['03.10.2015', 1, 0, 0, 20, 2.0],
        ['03.10.2015', 1, 0, 1, 100, 4.0],
        ['03.10.2015', 1, 0, 2, 1000, 1.0],
        ['03.10.2015', 1, 1, 0, 20, 5.0],
        ['03.10.2015', 1, 1, 1, 100, 2.0],
        ['03.10.2015', 1, 1, 2, 1000, 1.0],
    ]

    data += [
        ['04.10.2015', 1, 0, 0, 20, 2.0],
        ['04.10.2015', 1, 0, 1, 120, 2.0],
        ['04.10.2015', 1, 0, 2, 1000, 1.0],
        ['04.10.2015', 1, 1, 0, 20, 1.0],
        ['04.10.2015', 1, 1, 1, 120, 1.0],
        ['04.10.2015', 1, 1, 2, 1000, 1.0],
    ]

    data += [
        ['05.10.2015', 1, 0, 0, 20, 2.0],
        ['05.10.2015', 1, 0, 1, 100, 1.0],
        ['05.10.2015', 1, 0, 2, 1000, 1.0],
        ['05.10.2015', 1, 1, 0, 20, 5.0],
        ['05.10.2015', 1, 1, 1, 100, 2.0],
        ['05.10.2015', 1, 1, 2, 1000, 3.0],
    ]

    data += [
        ['06.10.2015', 1, 0, 0, 20, 2.0],
        ['06.10.2015', 1, 0, 1, 100, 1.0],
        ['06.10.2015', 1, 0, 2, 1000, 1.0],
        ['06.10.2015', 1, 1, 0, 20, 5.0],
        ['06.10.2015', 1, 1, 1, 110, 2.0],
        ['06.10.2015', 1, 1, 2, 1600, 10],
    ]
    return pd.DataFrame(data, columns=columns)
