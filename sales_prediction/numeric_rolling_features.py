def ndays_features(sales_df, col_name, num_days):
    groupby_obj = sales_df.groupby('shop_item_group')[col_name].rolling(num_days, min_periods=1)
    fmt = '{}_{}'.format(col_name, num_days) + 'day_{}'
    fns = [
        (groupby_obj.mean, (), fmt.format('mean')),
        (groupby_obj.std, (), fmt.format('std')),
        (groupby_obj.min, (), fmt.format('min')),
        (groupby_obj.max, (), fmt.format('max')),
        (groupby_obj.quantile, (0.1, ), fmt.format('qt_' + str(10))),
        (groupby_obj.quantile, (0.25, ), fmt.format('qt_' + str(25))),
        (groupby_obj.quantile, (0.5, ), fmt.format('qt_' + str(50))),
        (groupby_obj.quantile, (0.75, ), fmt.format('qt_' + str(75))),
        (groupby_obj.quantile, (0.95, ), fmt.format('qt_' + str(95))),
    ]
    return fns
