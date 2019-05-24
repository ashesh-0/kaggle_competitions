import pandas as pd


def add_lagged_features(df, feature_name, lags=[3, 6, 12]):
    df = df.reset_index()
    new_features = []
    for month_lag in lags:
        lag_f = df[['item_id', 'shop_id', feature_name, 'date_block_num']].copy()
        lag_f['date_block_num'] = lag_f['date_block_num'] + month_lag - 1
        new_fname = '{}_{}M'.format(feature_name, month_lag)
        new_features.append(new_fname)
        lag_f.rename({feature_name: new_fname}, inplace=True, axis=1)
        df = pd.merge(df, lag_f, how='left', on=['item_id', 'shop_id', 'date_block_num'])
        print('{} Month lagged feature computed.'.format(month_lag))

    df[new_features] = df[new_features].fillna(-10)
    return df.set_index('index')
