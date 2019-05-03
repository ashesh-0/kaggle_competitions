import pandas as pd
from numeric_features import get_y
from unittest.mock import patch


@patch('numeric_features.tqdm_notebook', side_effect=lambda x: x)
def test_get_y_should_return_correct_shape(mock_tqdm):
    columns = ['date', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        ['02.11.2013', 0, 1, 1.0],
        ['03.11.2013', 0, 1, 2.0],
        ['04.11.2013', 0, 1, 1.0],
        ['15.11.2013', 0, 1, 1.0],
        ['30.11.2013', 0, 1, 1.0],
        ['03.12.2013', 0, 1, 5.0],
        ['04.12.2013', 0, 1, 1.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    output_df = get_y(sales_df.copy())
    assert output_df.shape[0] == sales_df.shape[0]


@patch('numeric_features.tqdm_notebook', side_effect=lambda x: x)
def test_get_y_should_sum_one_month_single_item_shop(mock_tqdm):
    columns = ['date', 'shop_id', 'item_id', 'item_cnt_day']
    data = [
        ['02.11.2013', 0, 1, 1.0],
        ['03.11.2013', 0, 1, 2.0],
        ['04.11.2013', 0, 1, 1.0],
        ['15.11.2013', 0, 1, 1.0],
        ['30.11.2013', 0, 1, 1.0],
        ['03.12.2013', 0, 1, 5.0],
        ['05.12.2013', 0, 1, 1.0],
    ]
    output_data = [
        [6.0],
        [10.0],
        [8.0],
        [8.0],
        [7.0],
        [6.0],
        [1.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    sales_df.index += 10

    expected_df = pd.DataFrame(output_data, columns=['item_cnt_day'])
    expected_df.index = sales_df.index

    output_df = get_y(sales_df.copy())
    assert output_df.equals(expected_df['item_cnt_day'])


@patch('numeric_features.tqdm_notebook', side_effect=lambda x: x)
def test_get_y_should_sum_one_month_multiple_item_shop(mock_tqdm):
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
    output_data = [
        [6.0],
        [10.0],
        [8.0],
        [8.0],
        [7.0],
        [6.0],
        [1.0],
        [1.0],
        # Disjoint date
        [3.0],
        [1.0],
        # 2nd item
        [3.0],
        [2.0],
        # 2nd shop
        [6.0],
        [5.0],
    ]
    sales_df = pd.DataFrame(data, columns=columns)
    sales_df.index += 10

    expected_df = pd.DataFrame(output_data, columns=['item_cnt_day'])
    expected_df.index = sales_df.index

    output_df = get_y(sales_df.copy())
    print('')
    print(pd.concat([output_df.to_frame('output'), expected_df], axis=1))
    assert output_df.equals(expected_df['item_cnt_day'])
