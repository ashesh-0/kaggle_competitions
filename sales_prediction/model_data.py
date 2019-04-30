import pandas as pd
from numeric_features import get_y, get_numeric_X_df
from id_features import IdFeatures
from text_features import TextFeatures


def get_X_y(sales_df, items_df, category_en, shop_name_en):
    y_df = get_y(sales_df)
    print('Y fetched')
    X_df = get_X(sales_df, items_df, category_en, shop_name_en)

    # Retain common rows
    X_df = X_df.join(y_df[[]], how='inner')
    y_df = y_df.join(X_df[[]], how='inner')['item_cnt_day']
    # Order
    y_df = y_df.loc[X_df.index]
    return (X_df, y_df)


def get_X(sales_df, items_df, category_en, shop_name_en):

    X_df = get_numeric_X_df(sales_df)
    print('Numeric data fetched')

    # Adding text features.
    # Adding category id.
    X_df.index.name = 'index'
    X_df = X_df.reset_index()
    X_df = pd.merge(X_df, items_df[['item_id', 'item_category_id']], how='left', on='item_id')
    X_df = X_df.set_index('index')

    textf = TextFeatures(category_en, shop_name_en)
    shop_name_features_df = X_df['shop_id'].apply(textf.get_shop_feature_series)
    category_name_features_df = X_df['item_category_id'].apply(textf.get_category_feature_series)
    print('Text features fetched')
    X_df = pd.concat([X_df, shop_name_features_df, category_name_features_df], axis=1)
    print('Text features added')

    # Adding id features
    idf = IdFeatures(sales_df)
    X_df['item_id'] = X_df['item_id'].apply(idf.transform_item_id)
    X_df['shop_id'] = X_df['shop_id'].apply(idf.transform_shop_id)
    print('Id features added')
    return X_df


def get_X_test(sales_train_df, sales_test_df, category_en, shop_name_en, item_name_en):

    # Ensure that missing item_ids' are replaced.
    idf = IdFeatures(sales_train_df)
    idf.set_alternate_ids(item_name_en, sales_test_df)
    sales_test_df['item_id'] = sales_test_df['item_id'].apply(idf.transform_item_id_to_alternate_id)

    sales_test_df['date'] = '01.11.2015'
    sales_test_df['date_block_num'] = sales_train_df['date_block_num'].max() + 1

    # Ideally, this should not be needed. However, we need to set price and item_cnt_day.
    valid_prices = sales_train_df.groupby(['item_id', 'shop_id'])[['item_price']].last()
    valid_cnt = sales_train_df.groupby(['item_id', 'shop_id'])[['item_cnt_day']].mean()
    valid_dummy_values = pd.concat([valid_cnt, valid_prices], axis=1)

    sales_test_df = sales_test_df.reset_index()
    sales_test_df = pd.join([sales_test_df, valid_dummy_values], on=['item_id', 'shop_id'], how='left')
    sales_test_df = sales_test_df.set_index('index')
    recent_sales_df = sales_train_df[sales_train_df.date_f > pd.TimeStamp(year=2015, month=4)]

    subtract_index_offset = max(recent_sales_df.index) - (min(sales_test_df.index) - 1)
    recent_sales_df.index -= subtract_index_offset

    df = pd.concat([recent_sales_df, sales_test_df], axis=0)

    X_df = get_X(df)
    X_df = X_df.loc[sales_test_df.index]

    return X_df
