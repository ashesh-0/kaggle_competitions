import gc
import pandas as pd
from numeric_features import NumericFeatures, get_y, date_preprocessing
from id_features import IdFeatures
from text_features import TextFeatures


class ModelData:
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
        self._id_features = IdFeatures(self._sales_df, self._items_df, num_clusters=num_clusters)
        self._text_features = TextFeatures(category_en, shop_name_en)

    def get_train_X_y(self):
        X_df = self.preprocess_X(self._sales_df)
        y_df = get_y(self._sales_df)
        print('Y fetched')

        # Retain common rows
        X_df = X_df.join(y_df[[]], how='inner')
        y_df = y_df.join(X_df[[]], how='inner')['item_cnt_day']
        # Order
        y_df = y_df.loc[X_df.index]
        return (X_df, y_df)

    def preprocess_X(self, sales_df):
        X_df = self._numeric_features.get(sales_df)

        # Adding text features.
        shop_name_features_df = X_df['shop_id'].apply(self._text_features.get_shop_feature_series)
        category_name_features_df = X_df['item_category_id'].apply(self._text_features.get_category_feature_series)
        print('Text features fetched')
        X_df = pd.concat([X_df, shop_name_features_df, category_name_features_df], axis=1)
        print('Text features added')

        # Adding id features
        X_df['category_cluster'] = X_df['item_category_id'].map(
            self._id_features.transform_category_id_to_cluster_dict()).astype('int8')

        X_df['shop_cluster'] = X_df['shop_id'].map(self._id_features.transform_shop_id_to_cluster_dict()).astype('int8')

        X_df['item_id'] = X_df['item_id'].map(self._id_features.transform_item_id_dict()).astype('int32')
        X_df['shop_id'] = X_df['shop_id'].map(self._id_features.transform_shop_id_dict()).astype('int8')

        print('Id features added')
        return X_df

    def get_test_X(self, sales_test_df):
        gc.collect()
        # adding items_category_id to dataframe.
        item_to_cat_dict = self._items_df.set_index('item_id')['item_category_id'].to_dict()
        sales_test_df['item_category_id'] = sales_test_df.item_id.map(item_to_cat_dict)

        # Ensure that item_ids missing in train are replaced by nearby ids.
        self._id_features.set_alternate_ids(self._item_name_en, sales_test_df)
        assert self._id_features._item_id_alternate[83] != 83
        assert self._id_features._item_id_alternate[173] != 173

        sales_test_df['orig_item_id'] = sales_test_df['item_id']

        sales_test_df['item_id'] = sales_test_df['item_id'].map(
            self._id_features.transform_item_id_to_alternate_id_dict())

        assert sales_test_df[sales_test_df.item_id.isin([83, 173])].empty

        sales_test_df['date'] = '01.11.2015'
        sales_test_df['date_block_num'] = self._sales_df['date_block_num'].max() + 1

        # Ideally, this should not be needed. However, we need to set price and item_cnt_day.
        test_item_ids = sales_test_df.item_id.unique().tolist()
        sales_train_df = self._sales_df[self._sales_df.item_id.isin(test_item_ids)]

        # Ideally, this should not be needed. However, we need to set price and item_cnt_day.
        valid_prices = sales_train_df.groupby(['item_id', 'shop_id'])[['item_price']].last()
        valid_cnt = sales_train_df.groupby(['item_id', 'shop_id'])[['item_cnt_day']].mean()
        valid_dummy_values = pd.concat([valid_cnt, valid_prices], axis=1)

        # There are some pairs of shop_id, item_id which does not exist in train. For such, we will take mean value

        sales_test_df = sales_test_df.reset_index()
        sales_test_df = pd.merge(sales_test_df, valid_dummy_values, on=['item_id', 'shop_id'], how='left')

        #NOTE: need something better. Valid_dummy_values does not cover all pairs of item_id and shop_id.
        sales_test_df['item_cnt_day'] = sales_test_df['item_cnt_day'].fillna(sales_test_df['item_cnt_day'].mean())
        sales_test_df['item_price'] = sales_test_df['item_price'].fillna(sales_test_df['item_price'].mean())

        sales_test_df = sales_test_df.set_index('index')
        date_preprocessing(sales_test_df)

        recent_sales_df = sales_train_df[sales_train_df.date_f > pd.Timestamp(year=2015, month=6, day=1)]
        recent_sales_df = recent_sales_df.drop('shop_item_group', axis=1)

        subtract_index_offset = max(recent_sales_df.index) - (min(sales_test_df.index) - 1)
        recent_sales_df.index -= subtract_index_offset

        df = pd.concat([recent_sales_df, sales_test_df], axis=0, sort=False)

        del recent_sales_df, valid_cnt, valid_prices, valid_dummy_values

        print('Preprocessing X about to be done now.')
        X_df = self.preprocess_X(df)
        X_df = X_df.loc[sales_test_df.index]

        del df, sales_test_df
        gc.collect()

        return X_df
