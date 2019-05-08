import pandas as pd
import numpy as np


def monthly_features(sales_df):
    # monthly item sales
    monthly_item_sales_df = sales_df.groupby(['item_category_id', 'month', 'item_id',
                                              'shop_id'])['item_cnt_day'].sum().reset_index()

    # average monthly sales
    category_sales_grp = monthly_item_sales_df.groupby(['item_category_id', 'month'])['item_cnt_day']
    shop_sales_grp = monthly_item_sales_df.groupby(['shop_id', 'month'])['item_cnt_day']
    item_sales_grp = monthly_item_sales_df.groupby(['item_id', 'month'])['item_cnt_day']
    month_sales_grp = monthly_item_sales_df.groupby(['month'])['item_cnt_day']
    output = {}
    for groupby_obj, name in [
        (category_sales_grp, 'category'),
        (shop_sales_grp, 'shop'),
        (item_sales_grp, 'item'),
        (month_sales_grp, 'month'),
    ]:
        fmt = 'Monthly_{}'.format(name) + '_sales_cnt_{}'
        df1 = groupby_obj.mean().to_frame(fmt.format('mean'))
        df2 = groupby_obj.std().to_frame(fmt.format('std')).fillna(0)
        df3 = groupby_obj.min().to_frame(fmt.format('min'))
        df4 = groupby_obj.max().to_frame(fmt.format('max'))
        df5 = groupby_obj.quantile(0.1).to_frame(fmt.format('qt_' + str(10)))
        df6 = groupby_obj.quantile(0.25).to_frame(fmt.format('qt_' + str(25)))
        df7 = groupby_obj.quantile(0.5).to_frame(fmt.format('qt_' + str(50)))
        df8 = groupby_obj.quantile(0.75).to_frame(fmt.format('qt_' + str(75)))
        df9 = groupby_obj.quantile(0.95).to_frame(fmt.format('qt_' + str(95)))

        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1).astype('float32')
        # there could be some month for a particular item_id/shop_id/category_id which does not have any data point.
        # It means not a single item was bought and so item_cnt_day will be 0. So all features computed here will be 0.
        if isinstance(df.index, pd.MultiIndex):
            assert np.sum(df.isna().values) == 0
            df = df.unstack().fillna(0).stack().sort_index()

        output[name] = df

    return output


class MonthlyFeatures:
    def __init__(self, sales_df):
        assert 'item_category_id' in sales_df, 'item_category_id not found in sales_df'
        self._sales_df = sales_df
        self._monthly_features_dict = monthly_features(sales_df)

    def item_features(self, item_id_month: pd.Series):
        item_id = item_id_month.iloc[0]
        month = item_id_month.iloc[1]
        return self._monthly_features_dict['item'].loc[item_id, month]

    def shop_features(self, shop_id_month: pd.Series):
        shop_id = shop_id_month.iloc[0]
        month = shop_id_month.iloc[1]
        return self._monthly_features_dict['shop'].loc[shop_id, month]

    def category_features(self, category_id_month: pd.Series):
        category_id = category_id_month.iloc[0]
        month = category_id_month.iloc[1]
        return self._monthly_features_dict['category'].loc[category_id, month]

    def month_features(self, month: pd.Series):
        month = int(month.iloc[0])
        return self._monthly_features_dict['month'].loc[month]
