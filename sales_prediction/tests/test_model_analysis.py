from model_analysis import NewIdAnalysisValidationData
import pandas as pd
import numpy as np


def dummy_sales():
    columns = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']

    # Train
    data = [
        ['02.10.2013', 9, 0, 0, 10, 1.0],
        ['03.11.2013', 10, 0, 1, 100, 1.0],
        ['06.11.2013', 10, 1, 1, 100, 1.0],
    ]
    # item_id 2 is new in validation.
    # (1,0) shop_id, item_id tuple is missing in train.
    # Validation does not have (1,2) (shop_id, item_id) tuple.
    data += [
        ['03.12.2013', 11, 0, 0, 20, 2.0],
        ['03.12.2013', 11, 0, 1, 100, 1.0],
        ['03.12.2013', 11, 0, 2, 1000, 1.0],
        ['03.12.2013', 11, 1, 0, 20, 5.0],
        ['03.12.2013', 11, 1, 1, 100, 2.0],
    ]

    # redundant data.
    data += [
        ['02.01.2014', 12, 0, 0, 10, 1.0],
        ['03.01.2014', 12, 0, 1, 100, 1.0],
        ['06.01.2014', 12, 1, 1, 100, 1.0],
    ]
    return pd.DataFrame(data, columns=columns)


class TestNewIdAnalysisValidationData:
    def _initialize(self):
        sales_df = dummy_sales()
        y_df = pd.Series(np.random.rand(sales_df.shape[0]), index=sales_df.index)
        train_X = sales_df.iloc[:3].copy()

        val_X = sales_df.iloc[3:-3].copy()
        val_y = y_df.loc[val_X.index]

        # add missing tuple (1,2)
        new_index = val_X.index[-1] + 1
        val_X.loc[new_index] = ['01.12.2013', 11, 1, 2, 1000, 0.001]
        val_y.loc[new_index] = 0

        vdata = NewIdAnalysisValidationData(val_X, val_y, train_X, sales_df)
        return (vdata, val_X, val_y, train_X, sales_df)

    def test_get_new_item_ids(self):
        (vdata, val_X, val_y, train_X, sales_df) = self._initialize()
        new_items = vdata.get_new_item_ids()
        assert len(new_items) == 1 and new_items[0] == 2

    def test_get_new_item_based_validation_data(self):
        (vdata, val_X, val_y, train_X, sales_df) = self._initialize()

        f1_val_X, f1_val_y, percent1 = vdata.get_new_item_based_validation_data(new_ids=True)
        expected_X = val_X[val_X.item_id == 2]

        assert f1_val_X.equals(expected_X)
        assert f1_val_y.equals(val_y.loc[expected_X.index])
        assert percent1 == 100 * 2 / 6

        f0_val_X, f0_val_y, percent0 = vdata.get_new_item_based_validation_data(new_ids=False)
        expected_X = val_X[val_X.item_id != 2]
        assert f0_val_X.equals(expected_X)
        assert f0_val_y.equals(val_y.loc[expected_X.index])
        assert percent0 == 100 * 4 / 6

        assert val_X.equals(pd.concat([f1_val_X, f0_val_X], axis=0).sort_index())
        assert val_y.equals(pd.concat([f1_val_y, f0_val_y], axis=0).sort_index())

    def test_get_new_shop_based_validation_data(self):
        (vdata, val_X, val_y, train_X, sales_df) = self._initialize()

        f1_val_X, f1_val_y, percent1 = vdata.get_new_shop_based_validation_data(new_ids=True)

        assert f1_val_X.empty
        assert f1_val_y.empty
        assert percent1 == 0

        f0_val_X, f0_val_y, percent0 = vdata.get_new_shop_based_validation_data(new_ids=False)
        expected_X = val_X
        assert f0_val_X.equals(expected_X)
        assert f0_val_y.equals(val_y.loc[expected_X.index])
        assert percent0 == 100

    def test_get_new_shop_item_based_validation_data(self):
        (vdata, val_X, val_y, train_X, sales_df) = self._initialize()

        f1_val_X, f1_val_y, percent1 = vdata.get_new_shop_item_based_validation_data(new_ids=True)
        expected_filter = (val_X.item_id == 2) | ((val_X.item_id == 0) & (val_X.shop_id == 1))
        expected_X = val_X[expected_filter]

        assert f1_val_X.equals(expected_X)
        assert f1_val_y.equals(val_y.loc[expected_X.index])
        assert percent1 == 100 * 3 / 6

        f0_val_X, f0_val_y, percent0 = vdata.get_new_shop_item_based_validation_data(new_ids=False)
        expected_X = val_X[~expected_filter]
        assert f0_val_X.equals(expected_X)
        assert f0_val_y.equals(val_y.loc[expected_X.index])
        assert percent0 == 100 * 3 / 6

        assert val_X.equals(pd.concat([f1_val_X, f0_val_X], axis=0).sort_index())
        assert val_y.equals(pd.concat([f1_val_y, f0_val_y], axis=0).sort_index())
