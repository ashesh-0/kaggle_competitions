import numpy as np
import pandas as pd
from numeric_rolling_features import ndays_features
from numeric_features import basic_preprocessing

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


def test_ndays_features_should_sum_one_month_multiple_item_shop():
    columns = ['date', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        ['02.11.2013', 0, 1, 1.0],
        ['03.11.2013', 0, 1, 2.0],
        ['04.11.2013', 0, 1, 1.0],
        ['15.11.2013', 0, 1, 1.0],
        ['30.11.2013', 0, 1, 1.0],
        ['03.12.2013', 0, 1, 5.0],
        ['05.12.2013', 0, 1, 1.0],
        ['05.2.2014', 0, 1, 1.0],
        # Disjoint date
        ['03.10.2014', 0, 1, 2.0],
        ['14.10.2014', 0, 1, 1.0],
        # 2nd item.
        ['02.11.2013', 0, 2, 1.0],
        ['03.11.2013', 0, 2, 2.0],
        # 2nd shop
        ['30.11.2013', 1, 1, 1.0],
        ['03.12.2013', 1, 1, 5.0],
    ]
    fns = [np.sum, np.min, np.max]
    fn_names = ['sum', 'min', 'max', '1_q']
    output_data = [
        [1, 1, 1, 1],
        [3, 1, 2, 2],
        [4, 1, 2, 2],
        [5, 1, 2, 2],
        [6, 1, 2, 2],
        [10, 1, 5, 5],
        [8, 1, 5, 5],
        [1, 1, 1, 1],
        # Disjoint date
        [2, 2, 2, 2],
        [3, 1, 2, 2],
        # 2nd item
        [1, 1, 1, 1],
        [3, 1, 2, 2],
        # 2nd shop
        [1, 1, 1, 1],
        [6, 1, 5, 5],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    sales_df.index += 10
    basic_preprocessing(sales_df)

    expected_df = pd.DataFrame(
        output_data, columns=['item_cnt_day_{}d_{}'.format(30, f) for f in fn_names]).astype(float)
    expected_df.index = sales_df.index

    output_df = ndays_features(sales_df.copy(), 'item_cnt_day', 30, [1])
    print('')
    print(pd.concat([output_df, expected_df], axis=1, keys=['output', 'expected']))
    assert output_df.equals(expected_df)
