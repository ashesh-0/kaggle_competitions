import pandas as pd
from datetime import datetime

from numeric_utils import get_date_block_num
from numeric_features import NumericFeatures, get_y, date_preprocessing
from id_features import IdFeatures


class ModelData:
    EPSILON = 1e-4

    def __init__(self, sales_df, items_df, category_en, shop_name_en, item_name_en, num_clusters=(4, 4)):
        self._sales_df = sales_df
        self._items_df = items_df
        self._category_en = category_en
        self._shop_name_en = shop_name_en
        self._item_name_en = item_name_en

        # adding items_category_id to dataframe.
        item_to_cat_dict = self._items_df.set_index('item_id')['item_category_id'].to_dict()
        self._sales_df['item_category_id'] = self._sales_df.item_id.map(item_to_cat_dict)

        self._numeric_features = NumericFeatures(self._sales_df, self._items_df)

        # orig_item_id is needed in id_features.
        self._sales_df['orig_item_id'] = self._sales_df['item_id']
        self._id_features = IdFeatures(self._sales_df, self._items_df, num_clusters=num_clusters)
        # self._text_features = TextFeatures(category_en, shop_name_en)

    def get_train_X_y(self):
        print('Fetching X')
        X_df = self.get_X(self._sales_df)
        print('X fetched. Fetching y')
        y_df = get_y(self._sales_df).to_frame('item_cnt_month')
        print('Y fetched')

        # Retain common rows
        X_df = X_df.join(y_df[[]], how='inner')
        y_df = y_df.join(X_df[[]], how='inner')['item_cnt_month']
        # Order
        y_df = y_df.loc[X_df.index]
        return (X_df, y_df)

    def get_X(self, sales_df):
        X_df = self._numeric_features.get(sales_df)
        assert X_df.index.equals(sales_df.index)
        print('Numeric features fetched.')

        X_df['orig_item_id'] = sales_df['orig_item_id']
        X_df['date_block_num'] = sales_df['date_block_num']

        # # add first month related features for item_id
        X_df = self._id_features.get_fm_features(X_df, item_id_and_shop_id=False)
        # add first month related features for item_id and shop_id jointly
        X_df = self._id_features.get_fm_features(X_df, item_id_and_shop_id=True)

        # Adding text features.
        # shop_name_features_df = X_df['shop_id'].apply(self._text_features.get_shop_feature_series)
        # category_name_features_df = X_df['item_category_id'].apply(self._text_features.get_category_feature_series)
        # print('Text features fetched')
        # X_df = pd.concat([X_df, shop_name_features_df, category_name_features_df], axis=1)
        # print('Text features added')

        X_df['item_id_sorted'] = X_df['item_id'].map(
            self._id_features.transform_item_id_dict()).fillna(-1000).astype('int32')
        X_df['shop_id_sorted'] = X_df['shop_id'].map(
            self._id_features.transform_shop_id_dict()).fillna(-1000).astype('int32')

        print('Id features added')
        return X_df

    def get_test_X(self, sales_test_df, test_datetime: datetime, transform_missing_item_ids=False):

        item_id_original = sales_test_df['item_id'].copy()
        # adding items_category_id to dataframe.
        item_to_cat_dict = self._items_df.set_index('item_id')['item_category_id'].to_dict()
        sales_test_df['item_category_id'] = sales_test_df.item_id.map(item_to_cat_dict)

        sales_test_df['date'] = test_datetime.strftime('%d.%m.%Y')
        sales_test_df['date_block_num'] = get_date_block_num(test_datetime)
        sales_test_df['orig_item_id'] = sales_test_df['item_id']

        if transform_missing_item_ids:
            # Ensure that item_ids missing in train are replaced by nearby ids.
            self._id_features.set_alternate_ids(self._item_name_en, sales_test_df)
            sales_test_df['item_id'] = sales_test_df['item_id'].map(
                self._id_features.transform_item_id_to_alternate_id_dict())

        sales_test_df['item_cnt_day'] = 0
        sales_test_df['item_price'] = 0

        date_preprocessing(sales_test_df)
        assert sales_test_df.loc[item_id_original.index]['orig_item_id'].equals(item_id_original)

        test_dbn = get_date_block_num(test_datetime)
        recent_dbn = test_dbn - 5
        recent_sales_df = self._sales_df[(self._sales_df.date_block_num >= recent_dbn)
                                         & (self._sales_df.date_block_num < test_dbn)]

        recent_sales_df = recent_sales_df.drop('shop_item_group', axis=1)

        subtract_index_offset = max(recent_sales_df.index) - (min(sales_test_df.index) - 1)
        recent_sales_df.index -= subtract_index_offset

        df = pd.concat([recent_sales_df, sales_test_df], axis=0, sort=False)

        assert df.loc[item_id_original.index]['orig_item_id'].equals(item_id_original)

        print('Preprocessing X about to be done now.')
        X_df = self.get_X(df)

        X_df = X_df.loc[sales_test_df.index]

        assert X_df.loc[item_id_original.index]['orig_item_id'].equals(item_id_original)
        return X_df


if __name__ == '__main__':
    from constant_lists import CATEGORIES_EN, SHOPS_EN
    sales_df = pd.read_csv('data/sales_train.csv')
    sales_df.loc[sales_df['item_price'] < 0, 'item_price'] = 0

    items_df = pd.read_csv('data/items.csv')

    print(sales_df.head())
    print(items_df.head())

    md = ModelData(sales_df[sales_df.date_block_num > 26].copy(), items_df, CATEGORIES_EN, SHOPS_EN, [])
    X, y = md.get_train_X_y()
    print('Shape of X', X.shape)
    print(X.head())
