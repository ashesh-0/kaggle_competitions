from datetime import datetime, date
from numeric_utils import get_date_block_num


def test_get_date_block_num():
    assert 33 == get_date_block_num(date(2015, 10, 15))
    assert 33 == get_date_block_num(datetime(2015, 10, 15))
    assert 32 == get_date_block_num(datetime(2015, 9, 1))
    assert 0 == get_date_block_num(datetime(2013, 1, 1))
