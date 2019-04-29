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
    X_df = pd.merge(X_df, items_df[['item_id', 'item_category_id']], how='left', on='item_id')

    textf = TextFeatures(category_en, shop_name_en)
    shop_name_features_df = X_df['shop_id'].apply(textf.get_shop_feature_series)
    category_name_features_df = X_df['item_category_id'].apply(textf.get_shop_feature_series)
    print('Text features fetched')
    X_df = pd.concat([X_df, shop_name_features_df, category_name_features_df], axis=1)
    print('Text features added')

    # Adding id features
    idf = IdFeatures(sales_df)
    X_df['item_id'] = X_df['item_id'].apply(idf.transform_item_id)
    X_df['shop_id'] = X_df['shop_id'].apply(idf.transform_shop_id)
    print('Id features added')
    return X_df
