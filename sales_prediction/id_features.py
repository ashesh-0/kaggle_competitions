import pandas as pd

# DefaultValue = -1000
# class DefaultDict(dict):
#     def __init__(self, *args, **kwargs):
#         dict.__init__(self, *args, **kwargs)
#         self._default = DefaultValue

#     def __getitem__(self, idx):
#         return dict.get(self, idx, self._default)


class IdFeatures:
    ABSENT_ITEM_ID_VALUE = -1000

    def __init__(self, sales_df: pd.DataFrame, items_df: pd.DataFrame):

        self._sales_df = sales_df
        self._items_df = items_df

        # first time occuring features
        self._item_fm_df = None
        self._item_shop_fm_df = None

    def _fit_first_time_occuring_features(self):
        assert 'orig_item_id' in self._sales_df

        temp_df = self._sales_df[self._sales_df.item_cnt_day > 0][['orig_item_id', 'shop_id', 'date_block_num']]

        self._item_fm_df = temp_df.groupby(['orig_item_id'])['date_block_num'].min().to_frame('fm').reset_index()
        self._item_shop_fm_df = temp_df.groupby(['orig_item_id',
                                                 'shop_id'])['date_block_num'].min().to_frame('fm').reset_index()
        assert 'orig_item_id' in self._item_fm_df
        assert 'orig_item_id' in self._item_shop_fm_df

    def get_fm_features(self, df, item_id_and_shop_id=False):
        """
        Adds  first month features to df.
        df must have ['orig_item_id','shop_id','date_block_num'] columns
        """
        assert 'orig_item_id' in df
        if self._item_fm_df is None:
            self._fit_first_time_occuring_features()

        merge_df = self._item_shop_fm_df if item_id_and_shop_id else self._item_fm_df
        on_columns = ['orig_item_id', 'shop_id'] if item_id_and_shop_id else ['orig_item_id']
        f_nm_prefix = '_'.join(on_columns) + '_'
        df = pd.merge(df.reset_index(), merge_df, on=on_columns, how='left').set_index('index')

        old_col = f_nm_prefix + 'oldness'
        fm_col = f_nm_prefix + 'is_fm'

        df[old_col] = df['date_block_num'] - df['fm']
        # We will set oldness to 0 for which we don't have the data.
        df[old_col] = df[old_col].fillna(0).astype(int)

        df[fm_col] = df[old_col] == 0

        return df.drop('fm', axis=1)
