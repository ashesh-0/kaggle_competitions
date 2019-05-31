import pandas as pd
import numpy as np
from city_data import CITY_DATA, add_city_name
from sklearn.preprocessing import LabelEncoder


def add_city_data_features(X_df, shops_df):
    """
    Latitude, Longitude, Area and Importance of cities is added
    """
    shops_df = shops_df.copy()
    add_city_name(shops_df)
    shops_df['city_id'] = LabelEncoder().fit_transform(shops_df['city']).astype(np.int16)
    city_features = ['city_id']
    for key in ['lat', 'lon', 'importance', 'area']:
        feature = 'city_' + key
        shops_df[feature] = shops_df['city'].apply(lambda x: CITY_DATA[x][key] if x in CITY_DATA else -1).astype(
            np.float32)
        city_features.append(feature)

    X_df = X_df.reset_index()
    X_df = pd.merge(X_df, shops_df[city_features + ['shop_id']], how='left', on='shop_id')
    X_df.set_index('index', inplace=True)
    return X_df


if __name__ == '__main__':
    from constants import SALES_FPATH, SHOPS_FPATH
    sales_df = pd.read_csv(SALES_FPATH)
    shops_df = pd.read_csv(SHOPS_FPATH)

    print(add_city_data_features(sales_df, shops_df).head())
