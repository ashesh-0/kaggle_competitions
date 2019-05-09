"""
Contains a couple of functions which will help in analysis of performance of the model.
"""

import pandas as pd
from numeric_utils import get_datetime
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def get_metrics(model, input_df, y_df):
    pred_train = model.predict(input_df)
    pred_train[pred_train > 20] = 20
    mse = mean_squared_error(y_df.values, pred_train)
    r2 = r2_score(y_df.values, pred_train)
    return {'rmse': round(np.sqrt(mse), 2), 'r2': round(r2, 2)}


class NewIdAnalysisValidationData:
    def __init__(self, val_X, val_y, train_X, sales_df):
        self._val_X = val_X
        self._val_y = val_y
        self._train_X = train_X
        self._sales_df = sales_df

        self._train_sales_df = sales_df.loc[train_X.index]
        val_date_block_num = self._train_sales_df.date_block_num.max() + 1
        print('Validation started on ', get_datetime(val_date_block_num))

        self._val_sales_df = sales_df[sales_df.date_block_num == val_date_block_num]
        val_indices = set(val_X.index.tolist())
        val_sales_indices = set(self._val_sales_df.index.tolist())

        # validation X is a subset of sales_df and all other entries in validation X has 0 as target.
        assert val_sales_indices.issubset(val_indices)
        assert all(val_y[list(val_indices - val_sales_indices)].unique() == np.array([0]))

        # ensure that item_id and shop_ids are not reordered/changed
        assert self._val_X.loc[self._val_sales_df.index].item_id.equals(self._val_sales_df.item_id)
        assert self._val_X.loc[self._val_sales_df.index].shop_id.equals(self._val_sales_df.shop_id)

    def get_new_shop_ids(self):
        return list(set(self._val_sales_df.shop_id.unique()) - set(self._train_sales_df.shop_id.unique()))

    def get_new_item_ids(self):
        return list(set(self._val_sales_df.item_id.unique()) - set(self._train_sales_df.item_id.unique()))

    def _new_items_val_data_index_filter(self):
        item_ids = self.get_new_item_ids()
        return self._val_X.item_id.isin(item_ids)

    def _new_shops_val_data_index_filter(self):
        shop_ids = self.get_new_shop_ids()
        return self._val_X.shop_id.isin(shop_ids)

    def _new_shop_item_val_data_index_filter(self):
        validation_tuples_df = self._val_X[['item_id', 'shop_id']].apply(tuple, axis=1)

        validation_tuples = set(validation_tuples_df.values)
        train_tuples = set(self._train_sales_df[['item_id', 'shop_id']].apply(tuple, axis=1).values)
        new_tuples = list(validation_tuples - train_tuples)

        return validation_tuples_df.isin(new_tuples)

    def get_new_item_based_validation_data(self, new_ids=True):
        return self._filter_val_data('item_id', new_ids)

    def get_new_shop_based_validation_data(self, new_ids=True):
        return self._filter_val_data('shop_id', new_ids)

    def get_new_shop_item_based_validation_data(self, new_ids=True):
        return self._filter_val_data('shop_id_item_id', new_ids)

    def _filter_val_data(self, name, new_ids):
        if name == 'item_id':
            index_filter = self._new_items_val_data_index_filter()
        elif name == 'shop_id':
            index_filter = self._new_shops_val_data_index_filter()
        elif name == 'shop_id_item_id':
            index_filter = self._new_shop_item_val_data_index_filter()
        else:
            raise Exception('Incorrect name:' + name)

        if new_ids is False:
            index_filter = ~index_filter

        return (self._val_X[index_filter], self._val_y[index_filter], 100 * index_filter.sum() / self._val_X.shape[0])


def view_performance_on_new_ids(model, vdata_obj):
    views = {
        'item': vdata_obj.get_new_item_based_validation_data,
        'shop': vdata_obj.get_new_shop_based_validation_data,
        'shop_item': vdata_obj.get_new_shop_item_based_validation_data,
    }
    for name, fn in views.items():
        val_X_df, val_y_df, percent = fn(new_ids=True)
        m = get_metrics(model, val_X_df, val_y_df)
        print('[NEW]--[{}] Percent:{:.2f}% \tRMSE:{} \tR2:{}'.format(name, percent, m['rmse'], m['r2']))

        val_X_df, val_y_df, percent = fn(new_ids=False)
        m = get_metrics(model, val_X_df, val_y_df)
        print('[OLD]--[{}] Percent:{:.2f}% \tRMSE:{} \tR2:{}'.format(name, percent, m['rmse'], m['r2']))


def get_Xy_df_with_ids(model, val_X_df, val_y_df, sales_df, items_df: pd.DataFrame, train_X_df: pd.DataFrame):
    """
    Returns a dataframe which has item_id, shop_id, item_category_id, prediction and actual with index being same as of
    sales_df.
    It also returns an index which is of those entries which belonged to the train data (other entries are generated).
    """
    train_sales_df = sales_df.loc[train_X_df.index]
    val_date_block_num = train_sales_df.date_block_num.max() + 1
    print('Validation started on ', get_datetime(val_date_block_num))

    val_sales_df = sales_df[sales_df.date_block_num == val_date_block_num]
    val_indices = set(val_X_df.index.tolist())
    val_sales_indices = set(val_sales_df.index.tolist())

    # validation X is a subset of sales_df and all other entries in validation X has 0 as target.
    assert val_sales_indices.issubset(val_indices)
    assert all(val_y_df[list(val_indices - val_sales_indices)].unique() == np.array([0]))

    val_X_df = pd.merge(
        val_X_df[['item_id', 'shop_id']].reset_index(), items_df, on='item_id', how='left').set_index('index')

    prediction_df = pd.Series(model.predict(val_X_df.values), index=val_X_df.index)
    val_X_df['prediction'] = prediction_df
    val_X_df['actual'] = val_y_df
    val_X_df['square_error'] = (val_X_df['actual'] - val_X_df['prediction']).pow(2)
    return (val_X_df, val_sales_df.index)


# performance on new shop_id
# performance on new item_id
# performance on new (shop_id, item_id) tuple.
# performance on category_id
# performance on shop_id
# performance as a fn of number of monthly sales.
# distribution of the above in train and some validation.
