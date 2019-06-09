import pandas as pd
from tqdm import tqdm_notebook

# df = corr.stack()
# df.drop([(c,c) for c in X_df.columns],inplace=True)

# columns_to_remove = []
# identical_features = df[df.abs() > 0.995]
# for index in identical_features.index:
# #     print(index)
#     if index[0] in columns_to_remove or index[1] in columns_to_remove:
#         continue
#     columns_to_remove.append(index[1])


def last_traded_column_name(id_col):
    return 'last_traded_{}'.format(id_col)


def first_month_column_name(id_col):
    return 'first_month_{}'.format(id_col)


def oldness_column_name(id_col):
    return 'oldness_{}'.format(id_col)


def last_traded_feature(id_dbn_df, id_col):
    id_dbn_df.sort_values('date_block_num', inplace=True)
    id_dbn_df = id_dbn_df.groupby(['date_block_num', id_col])[['item_cnt_day']].first().reset_index()
    feature = id_dbn_df.groupby(id_col)['date_block_num'].rolling(
        2, min_periods=2).apply(
            lambda ser: ser.iloc[-2], raw=False)
    feature = feature.to_frame('last_date_block_num')
    feature = feature.reset_index(level=0)
    feature['date_block_num'] = id_dbn_df['date_block_num']

    last_month_feature = id_dbn_df.groupby(id_col)['date_block_num'].max().to_frame('last_date_block_num').reset_index()
    last_month_feature['date_block_num'] = last_month_feature['last_date_block_num'] + 1
    feature = pd.concat([feature, last_month_feature]).sort_values('date_block_num')

    feature = feature.set_index([id_col, 'date_block_num'])[['last_date_block_num']].unstack().T
    feature = feature.fillna(method='ffill').T.stack().reset_index()
    feature[last_traded_column_name(id_col)] = feature['date_block_num'] - feature['last_date_block_num']
    feature.drop('last_date_block_num', axis=1, inplace=True)
    return feature


def oldness_feature(id_dbn_df, id_col):
    first_month_df = id_dbn_df.groupby(id_col)['date_block_num'].min()

    feature = pd.DataFrame(
        [list(range(35))] * first_month_df.shape[0], index=first_month_df.index, columns=list(range(35))).astype(
            np.int16)
    feature.columns.name = 'date_block_num'
    feature.index.name = id_col

    feature = feature.subtract(first_month_df, axis=0)
    feature[feature < 0] = -10

    oldness_col = oldness_column_name(id_col)
    firstm_col = first_month_column_name(id_col)

    feature = feature.stack().to_frame(oldness_col).reset_index()
    feature[firstm_col] = feature[oldness_col] == 0

    return feature[[id_col, 'date_block_num', oldness_col, firstm_col]]


class TradingDurationBasedFeatures:
    def __init__(self, sales_df, items_df):
        self._sales = sales_df[sales_df.item_cnt_day != 0].copy()
        self._sales['item_shop_id'] = self._sales['item_id'] * 100 + self._sales['shop_id']

        if 'item_category_id' not in self._sales:
            self._sales = pd.merge(self._sales, items_df[['item_id', 'item_category_id']], how='left', on='item_id')

        self._sales['shop_category_id'] = self._sales['shop_id'] * 100 + self._sales['item_category_id']
        self._column_ids = ['item_id', 'shop_id', 'item_category_id', 'shop_category_id', 'item_shop_id']

        # Oldness based features
        self._oldness_dict = {}
        # Last traded features. We need to join them with X_df on (id, date_block_num). We need to fillna.
        self._last_traded_dict = {}

        for col_id in tqdm_notebook(self._column_ids):
            col_id_dbn_df = self._sales.groupby([col_id, 'date_block_num'])['item_cnt_day'].first().reset_index()
            self._last_traded_dict[col_id] = last_traded_feature(col_id_dbn_df, col_id)
            self._oldness_dict[col_id] = oldness_feature(col_id_dbn_df, col_id)
            print(col_id, 'computed')

    def transform(self, X_df):
        assert len(set(['item_id', 'shop_id', 'item_category_id']) - set(X_df.columns.tolist())) == 0
        X_df = X_df.reset_index()

        assert 'shop_category_id' not in X_df
        assert 'item_shop_id' not in X_df

        X_df['item_shop_id'] = X_df['item_id'] * 100 + X_df['shop_id']
        X_df['shop_category_id'] = X_df['shop_id'] * 100 + X_df['item_category_id']

        for col_id in tqdm_notebook(self._column_ids):
            X_df = pd.merge(X_df, self._last_traded_dict[col_id], how='left', on=['date_block_num', col_id])
            X_df[last_traded_column_name(col_id)] = X_df[last_traded_column_name(col_id)].fillna(-10)

            X_df = pd.merge(X_df, self._oldness_dict[col_id], how='left', on=['date_block_num', col_id])

            # If id is not found, it means that it is new.
            X_df[first_month_column_name(col_id)] = X_df[first_month_column_name(col_id)].fillna(True)
            X_df[oldness_column_name(col_id)] = X_df[oldness_column_name(col_id)].fillna(0)

        X_df.set_index('index', inplace=True)
        X_df.drop(['item_shop_id', 'shop_category_id'], axis=1, inplace=True)

        return X_df
