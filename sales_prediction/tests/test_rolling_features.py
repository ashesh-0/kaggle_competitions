import pandas as pd
from numeric_rolling_features import nmonths_features
from numeric_features import basic_preprocessing

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


def test_nmonths_features_should_sum_one_month_multiple_item_shop():
    columns = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        [11, 0, 1, 1.0],
        [12, 0, 1, 2.0],
        [13, 0, 1, 1.0],
        [14, 0, 1, 1.0],
        [15, 0, 1, 1.0],
        [16, 0, 1, 5.0],
        # Disjoint date
        [20, 0, 1, 2.0],
        [33, 0, 1, 1.0],
        # 2nd item.
        [12, 0, 2, 1.0],
        [13, 0, 2, 2.0],
        # 2nd shop
        [12, 1, 1, 1.0],
        [13, 1, 1, 5.0],
        [14, 1, 1, 5.0]
    ]
    fn_names = ['sum', 'min', 'max', '1_q']
    output_data = [
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [3, 1, 2, 2],
        [4, 1, 2, 2],
        [5, 1, 2, 2],
        [5, 1, 2, 2],
        # Disjoint date
        [5, 5, 5, 5],
        [0, 0, 0, 0],
        # 2nd item
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        # 2nd shop
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [6, 1, 5, 5],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    sales_df.index += 10
    basic_preprocessing(sales_df)

    expected_df = pd.DataFrame(
        output_data, columns=['item_cnt_day_{}M_{}'.format(4, f) for f in fn_names]).astype(float)
    expected_df.index = sales_df.index

    output_df = nmonths_features(sales_df.copy(), 'item_cnt_day', 4, [1])
    print('')
    print(pd.concat([output_df, expected_df], axis=1, keys=['output', 'expected']))
    assert output_df.equals(expected_df)
