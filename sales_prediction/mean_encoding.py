from sklearn.model_selection import KFold


class MeanEncoding:
    MAX_SHOP_ID = 59
    MAX_ITEM_CATEGORY_ID = 83
    MAX_ITEM_ID = 22_169

    def __init__(self, train_X_df, train_y_df, val_X_df, val_y_df):

        self._X = train_X_df
        self._y = train_y_df

        self._val_X = val_X_df
        self._val_y = val_y_df

        self.validations(self._X)
        self.validations(self._val_X)

        self._item_encoding_map = {}
        self._shop_encoding_map = {}
        self._item_category_encoding_map = {}
        self._item_shop_encoding_map = {}
        self._shop_category_encoding_map = {}
        self._month_encoding_map = {}

        self.fit_all()

        self.transform(self._val_X)

    def transform(self, df):
        self.transform_item_id(df)
        self.transform_shop_id(df)
        self.transform_category_id(df)
        self.transform_item_shop_id(df)
        self.transform_shop_category_id(df)
        self.transform_month(df)

    def validations(self, X):
        assert 'item_id' in X.columns
        assert 'shop_id' in X.columns
        assert 'item_category_id' in X.columns
        assert 'month' in X.columns

    def get_train_data(self, n_splits=5):
        X = self._X.copy()
        X['target'] = self._y

        # needed for (item_id,shop_id) encoding.
        X['item_shop_id'] = self._get_item_shop_id(X)
        X['shop_category_id'] = self._get_shop_category_id(X)

        month_enc_c = 'month_mean_enc'
        item_enc_c = 'item_id_mean_enc'
        shop_enc_c = 'shop_id_mean_enc'
        item_category_enc_c = 'item_category_id_mean_enc'
        item_shop_enc_c = 'item_shop_id_mean_enc'
        shop_category_enc_c = 'shop_category_id_mean_enc'

        X[item_enc_c] = 0
        X[shop_enc_c] = 0
        X[item_category_enc_c] = 0
        X[item_shop_enc_c] = 0
        X[shop_category_enc_c] = 0
        X[month_enc_c] = 0

        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X):
            indices = X.index[test_index]
            X_tr = X.iloc[train_index]

            item_id_encoding = self._fit(X_tr, 'item_id', MeanEncoding.MAX_ITEM_ID)
            shop_id_encoding = self._fit(X_tr, 'shop_id', MeanEncoding.MAX_SHOP_ID)
            item_category_id_encoding = self._fit(X_tr, 'item_category_id', MeanEncoding.MAX_ITEM_CATEGORY_ID)
            item_shop_id_encoding = self._fit(X_tr, 'item_shop_id',
                                              MeanEncoding.MAX_ITEM_ID * 100 + MeanEncoding.MAX_SHOP_ID)
            shop_category_id_encoding = self._fit(X_tr, 'shop_category_id',
                                                  MeanEncoding.MAX_SHOP_ID * 100 + MeanEncoding.MAX_ITEM_CATEGORY_ID)

            month_encoding = self._fit(X_tr, 'month', 12)

            X.loc[indices, month_enc_c] = X.loc[indices, 'month'].map(month_encoding)
            X.loc[indices, item_enc_c] = X.loc[indices, 'item_id'].map(item_id_encoding)
            X.loc[indices, shop_enc_c] = X.loc[indices, 'shop_id'].map(shop_id_encoding)
            X.loc[indices, item_category_enc_c] = X.loc[indices, 'item_category_id'].map(item_category_id_encoding)
            X.loc[indices, item_shop_enc_c] = X.loc[indices, 'item_shop_id'].map(item_shop_id_encoding)
            X.loc[indices, shop_category_enc_c] = X.loc[indices, 'shop_category_id'].map(shop_category_id_encoding)

        return X.drop(['target', 'item_shop_id', 'shop_category_id'], axis=1), self._y

    def fit_all(self):
        X_df = self._X.copy()
        X_df['target'] = self._y
        self.fit_item_category_id(X_df)
        print('[ME] category done')
        self.fit_item_id(X_df)
        print('[ME] item done')
        self.fit_shop_id(X_df)
        print('[ME] shop done')
        self.fit_item_shop_id(X_df)
        print('[ME] item shop done')
        self.fit_shop_category_id(X_df)
        print('[ME] shop category done')
        self.fit_month(X_df)
        print('[ME] month done')

    def _fit(self, X_df, id_column, max_id):
        item_encoding_map = X_df.groupby([id_column])['target'].mean().to_dict()
        for id_ in range(max_id + 1):
            item_encoding_map[id_] = item_encoding_map.get(id_, 0)
        return item_encoding_map

    def fit_month(self, X_df):
        self._month_encoding_map = self._fit(X_df, 'month', 12)

    def fit_item_id(self, X_df):
        self._item_encoding_map = self._fit(X_df, 'item_id', MeanEncoding.MAX_ITEM_ID)

    def fit_shop_id(self, X_df):
        self._shop_encoding_map = self._fit(X_df, 'shop_id', MeanEncoding.MAX_SHOP_ID)

    def fit_item_category_id(self, X_df):
        self._item_category_encoding_map = self._fit(X_df, 'item_category_id', MeanEncoding.MAX_ITEM_CATEGORY_ID)

    def _get_item_shop_id(self, df):
        return df['item_id'] * 100 + df['shop_id']

    def _get_shop_category_id(self, df):
        return df['shop_id'] * 100 + df['item_category_id']

    def fit_shop_category_id(self, X_df):
        assert MeanEncoding.MAX_ITEM_CATEGORY_ID < 100
        X_df['shop_category_id'] = self._get_shop_category_id(X_df)

        self._shop_category_encoding_map = self._fit(X_df, 'shop_category_id',
                                                     100 * MeanEncoding.MAX_ITEM_ID + MeanEncoding.MAX_SHOP_ID)

        X_df.drop('shop_category_id', axis=1, inplace=True)

    def fit_item_shop_id(self, X_df):
        assert MeanEncoding.MAX_SHOP_ID < 100
        X_df['item_shop_id'] = self._get_item_shop_id(X_df)

        self._item_shop_encoding_map = self._fit(X_df, 'item_shop_id',
                                                 100 * MeanEncoding.MAX_ITEM_ID + MeanEncoding.MAX_SHOP_ID)

        X_df.drop('item_shop_id', axis=1, inplace=True)

    def transform_item_shop_id(self, df):
        item_shop_id = self._get_item_shop_id(df)
        df['item_shop_id_mean_enc'] = item_shop_id.map(self._item_shop_encoding_map)

    def transform_shop_category_id(self, df):
        shop_category_id = self._get_shop_category_id(df)
        df['shop_category_id_mean_enc'] = shop_category_id.map(self._shop_category_encoding_map)

    def transform_item_id(self, df):
        df['item_id_mean_enc'] = df['item_id'].map(self._item_encoding_map)

    def transform_shop_id(self, df):
        df['shop_id_mean_enc'] = df['shop_id'].map(self._shop_encoding_map)

    def transform_category_id(self, df):
        df['item_category_id_mean_enc'] = df['item_category_id'].map(self._item_category_encoding_map)

    def transform_month(self, df):
        df['month_mean_enc'] = df['month'].map(self._month_encoding_map)

    def get_val_data(self):
        return (self._val_X, self._val_y)
