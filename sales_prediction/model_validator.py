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

    def get_train_val_data(self):
        train_X_df, train_y_df = self.get_train_data()
        val_X_df, val_y_df = self.get_val_data()
        return (train_X_df, train_y_df, val_X_df, val_y_df)

    def get_val_data(self):

        val_date_block_num = self._train_end_date_block_num - self._skip_last_n_months
        val_X_df = self._sales_df[self._sales_df.date_block_num == val_date_block_num][['item_id',
                                                                                        'shop_id']].reset_index()
        val_X_df = val_X_df.groupby(['item_id', 'shop_id']).first().reset_index().set_index('index')
        val_X_df = val_X_df.sort_index().astype(int)
        val_y_df = self._y_df.loc[val_X_df.index].copy()

        return (val_X_df, val_y_df)
