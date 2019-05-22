import pandas as pd


def lagged_features(df, feature_name, lags=[3, 6, 12]):
    df = df.reset_index()
    for month_lag in lags:
        lag_f = df[['item_id', 'shop_id', feature_name, 'date_block_num']].copy()
        lag_f['date_block_num'] = lag_f['date_block_num'] + month_lag - 1
        lag_f.rename({feature_name: '{}_{}M'.format(feature_name, month_lag)}, inplace=True, axis=1)
        df = pd.merge(df, lag_f, how='left', on=['item_id', 'shop_id', 'date_block_num'])
        print('{} Month lagged feature computed.', month_lag)

    return df.set_index('index')
