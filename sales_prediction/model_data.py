from model_validator import ModelValidator
from rolling_mean_encoding import rolling_mean_encoding
import pandas as pd
from datetime import datetime
from constants import (DATA_FPATH, TEST_LIKE_SALES_FPATH, SALES_FPATH, ITEMS_FPATH, SHOPS_FPATH, TEST_SALES_FPATH,
                       ITEM_CATEGORIES_FPATH)

from numeric_utils import get_date_block_num
from numeric_features import NumericFeatures, get_y, date_preprocessing
from id_features import IdFeatures
from lagged_features import add_lagged_features
from city_data_features import add_city_data_features
from duration_features import TradingDurationBasedFeatures


class ModelData:
    EPSILON = 1e-4

    def __init__(self, sales_df, items_df, shops_df):
        self._sales_df = sales_df
        self._items_df = items_df
        self._shops_df = shops_df

        # adding items_category_id to dataframe.
        item_to_cat_dict = self._items_df.set_index('item_id')['item_category_id'].to_dict()
        self._sales_df['item_category_id'] = self._sales_df.item_id.map(item_to_cat_dict)

        self._numeric_features = NumericFeatures(self._sales_df, self._items_df)

        # orig_item_id is needed in id_features.
        self._sales_df['orig_item_id'] = self._sales_df['item_id']
        self._id_features = IdFeatures(self._sales_df, self._items_df)
        self._duration_features = TradingDurationBasedFeatures(self._sales_df, self._items_df)

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

        print('Id features added')

        # lagged features added
        X_df = add_lagged_features(X_df, 'item_cnt_day_1M_sum')
        print('Lagged features added')

        # City related features like area, coordinate, city_id
        X_df = add_city_data_features(X_df, self._shops_df)
        print('City data features added')

        # Duration features
        assert not X_df.isna().any().any()
        X_df = self._duration_features.transform(X_df)
        print('Duration features added')
        assert not X_df.isna().any().any()

        return X_df

    def get_test_X(self, sales_test_df, test_datetime: datetime):

        item_id_original = sales_test_df['item_id'].copy()
        # adding items_category_id to dataframe.
        item_to_cat_dict = self._items_df.set_index('item_id')['item_category_id'].to_dict()
        sales_test_df['item_category_id'] = sales_test_df.item_id.map(item_to_cat_dict)

        sales_test_df['date'] = test_datetime.strftime('%d.%m.%Y')
        sales_test_df['date_block_num'] = get_date_block_num(test_datetime)
        sales_test_df['orig_item_id'] = sales_test_df['item_id']

        sales_test_df['item_cnt_day'] = 0
        sales_test_df['item_price'] = 0

        date_preprocessing(sales_test_df)
        assert sales_test_df.loc[item_id_original.index]['orig_item_id'].equals(item_id_original)

        test_dbn = get_date_block_num(test_datetime)
        recent_sales_df = self._sales_df[self._sales_df.date_block_num < test_dbn]

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


def mean_encoding_preprocessing(sales):
    test_sales_df = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv', index_col=0)
    test_sales_df['date_block_num'] = 34
    test_sales_df['item_cnt_day'] = 0.33
    index_offset = sales.index.max() + 1
    print('Offsetting test sales index by ', index_offset)
    test_sales_df.index += index_offset
    combined_sales_df = pd.concat(
        [sales[['date_block_num', 'item_id', 'shop_id', 'item_cnt_day']], test_sales_df], sort=True)
    combined_sales_df = pd.merge(
        combined_sales_df.reset_index(), items[['item_id', 'item_category_id']], on='item_id',
        how='left').set_index('index')

    combined_sales_df = rolling_mean_encoding(combined_sales_df)
    train_mean_encoding = combined_sales_df.loc[sales.index].copy()
    test_mean_encoding = combined_sales_df.loc[test_sales_df.index].copy()
    test_mean_encoding.index = test_mean_encoding.index - (sales.index.max() + 1)
    assert test_mean_encoding.shape[0] + train_mean_encoding.shape[0] == combined_sales_df.shape[0]
    assert (test_mean_encoding.date_block_num == 34).all()

    return (train_mean_encoding, test_mean_encoding)


def get_val_dfs(skip_last_n_months: int):
    """
    skip_last_n_months decides how many months prior to Nov 2015 to take as the validation data.
    """
    year = (1 + 33 - skip_last_n_months) // 12 + 2013
    month = (1 + 33 - skip_last_n_months) % 12

    mv = ModelValidator(sales, X_df, y_df, skip_last_n_months=skip_last_n_months)
    val_X_ids, val_y_df = mv.get_val_data()
    print('Before get_test_X', val_X_ids.shape, val_y_df.shape)
    val_X_df = md.get_test_X(val_X_ids, datetime(year, month, 1))
    assert val_X_df.index.equals(val_X_ids.index)
    print('After get_test_X', val_X_df.shape)
    return (val_X_df, val_y_df)


if __name__ == '__main__':
    sales = pd.read_hdf(TEST_LIKE_SALES_FPATH, 'df')
    items = pd.read_csv(ITEMS_FPATH)
    shops = pd.read_csv(SHOPS_FPATH)
    categ = pd.read_csv(ITEM_CATEGORIES_FPATH)
    test = pd.read_csv(TEST_SALES_FPATH)

    # Cleaning up
    sales.loc[sales['item_price'] < 0, 'item_price'] = 0
    sales.loc[sales['item_cnt_day'] > 300, 'item_cnt_day'] = 300
    sales.loc[sales['item_cnt_day'] < 0, 'item_cnt_day'] = 0

    # all features other than mean encoding is handled through this class.
    md = ModelData(sales, items)

    # get train data.
    X_df, y_df = md.get_train_X_y()

    # Get test data.
    test_df = md.get_test_X(test, datetime(2015, 11, 1))

    # Get mean encodings
    train_mean_encoding, test_mean_encoding = mean_encoding_preprocessing(sales)

    test_mean_encoding.drop(
        ['date_block_num', 'item_cnt_day', 'item_id', 'shop_id', 'item_category_id'], axis=1, inplace=True)
    train_mean_encoding.drop(
        ['date_block_num', 'item_cnt_day', 'item_id', 'shop_id', 'item_category_id'], axis=1, inplace=True)
    del combined_sales_df

    X_df = pd.concat([X_df, train_mean_encoding], axis=1)
    test_df = pd.concat([test_df, test_mean_encoding], axis=1)

    # Saving train and test data to disk
    X_df.to_hdf(DATA_FPATH, 'X')
    y_df.to_hdf(DATA_FPATH, 'y')
    test_df.to_hdf(DATA_FPATH, 'test_X')
