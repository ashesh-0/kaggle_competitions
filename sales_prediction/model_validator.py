import pandas as pd
import numpy as np
from numeric_utils import get_items_in_market, get_shops_in_market


class ModelValidator:
    """
    A class to return a validation set in the same way it is present in test set.
    """

    def __init__(self, sales_df, X_df, y_df, skip_last_n_months: int = 0):
        self._sales_df = sales_df
        self._X_df = X_df
        self._y_df = y_df

        self._train_end_date_block_num = self._sales_df.date_block_num.max()
        self._skip_last_n_months = skip_last_n_months

    def get_train_data(self):
        val_date_block_num = self._train_end_date_block_num - self._skip_last_n_months
        train_X_df = self._X_df.loc[self._sales_df[self._sales_df.date_block_num < val_date_block_num].index]
        train_y_df = self._y_df.loc[train_X_df.index]
        return (train_X_df, train_y_df)

    def get_new_items_shops_info(self):
        val_date_block_num = self._train_end_date_block_num - self._skip_last_n_months
        train_X_df = self._sales_df[self._sales_df.date_block_num < val_date_block_num]
        val_X_df = self._sales_df[self._sales_df.date_block_num == val_date_block_num]

        output = {}
        output['new_items'] = list(set(val_X_df.item_id.unique()) - set(train_X_df.item_id.unique()))
        output['new_shops'] = list(set(val_X_df.shop_id.unique()) - set(train_X_df.shop_id.unique()))
        val_tuples = set(val_X_df[['item_id', 'shop_id']].apply(tuple, axis=1).values)
        train_tuples = set(train_X_df[['item_id', 'shop_id']].apply(tuple, axis=1).values)
        output['new_item_shops'] = list(val_tuples - train_tuples)
        return output

    def get_train_val_data(self):
        train_X_df, train_y_df = self.get_train_data()
        val_X_df, val_y_df = self.get_val_data()
        return (train_X_df, train_y_df, val_X_df, val_y_df)

    def get_val_data(self):

        val_date_block_num = self._train_end_date_block_num - self._skip_last_n_months
        val_X_df = self._sales_df[self._sales_df.date_block_num == val_date_block_num][['item_id',
                                                                                        'shop_id']].reset_index()
        val_X_df = val_X_df.groupby(['item_id', 'shop_id']).first().reset_index().set_index('index')
        val_y_df = self._y_df.loc[val_X_df.index].copy()

        item_ids = get_items_in_market(self._sales_df, val_date_block_num)
        shop_ids = get_shops_in_market(self._sales_df, val_date_block_num)

        existing_ids = set(val_X_df[['item_id', 'shop_id']].apply(tuple, axis=1).values)

        extra_ids = np.zeros((len(item_ids) * len(shop_ids) - len(existing_ids), 2))
        index = 0
        for item_id in item_ids:
            for shop_id in shop_ids:
                if (item_id, shop_id) in existing_ids:
                    continue
                extra_ids[index, :] = [item_id, shop_id]
                index += 1

        other_val_X_df = pd.DataFrame(extra_ids, columns=['item_id', 'shop_id'])
        other_val_X_df.index += val_X_df.index.max() + 1
        other_val_y_df = pd.Series(np.zeros(other_val_X_df.shape[0]), index=other_val_X_df.index)

        val_X_df = pd.concat([val_X_df, other_val_X_df])
        val_y_df = pd.concat([val_y_df, other_val_y_df])

        val_X_df = val_X_df.sort_index().astype(int)
        val_y_df = val_y_df.loc[val_X_df.index]

        return (val_X_df, val_y_df)
