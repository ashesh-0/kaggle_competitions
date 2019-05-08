import pandas as pd
import numpy as np
from datetime import datetime, date
from numeric_utils import get_items_in_market, get_date_block_num, get_shops_in_market


def test_get_date_block_num():
    assert 33 == get_date_block_num(date(2015, 10, 15))
    assert 33 == get_date_block_num(datetime(2015, 10, 15))
    assert 32 == get_date_block_num(datetime(2015, 9, 1))
    assert 0 == get_date_block_num(datetime(2013, 1, 1))


def test_get_items_in_market():
    df = pd.DataFrame(
        [
            [1, 1],
            [2, 1],
            [2, 2],
            [3, 5],
            [3, 4],
        ], columns=['item_id', 'date_block_num'])
    assert all(np.array([1, 2]) == np.array(get_items_in_market(df, 1)))
    assert all(np.array([2]) == np.array(get_items_in_market(df, 2)))
    assert all(np.array([]) == np.array(get_items_in_market(df, 3)))
    assert all(np.array([3]) == np.array(get_items_in_market(df, 4)))
    assert all(np.array([3]) == np.array(get_items_in_market(df, 5)))


def test_get_shops_in_market():
    df = pd.DataFrame(
        [
            [1, 1],
            [2, 1],
            [2, 2],
            [3, 5],
            [3, 4],
        ], columns=['shop_id', 'date_block_num'])
    assert all(np.array([1, 2]) == np.array(get_shops_in_market(df, 1)))
    assert all(np.array([2]) == np.array(get_shops_in_market(df, 2)))
    assert all(np.array([]) == np.array(get_shops_in_market(df, 3)))
    assert all(np.array([3]) == np.array(get_shops_in_market(df, 4)))
    assert all(np.array([3]) == np.array(get_shops_in_market(df, 5)))
