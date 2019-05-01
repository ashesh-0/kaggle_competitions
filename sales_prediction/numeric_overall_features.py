import pandas as pd


def overall_features(sales_df, items_df):
    merged_df = pd.merge(sales_df, items_df, on='item_id', how='left')
    # monthly item sales
    monthly_item_sales_df = merged_df.groupby(['item_category_id', 'month', 'item_id',
                                               'shop_id'])['item_cnt_day'].sum().reset_index()
    category_sales_grp = monthly_item_sales_df.groupby(['item_category_id'])['item_cnt_day']
    shop_sales_grp = monthly_item_sales_df.groupby(['shop_id'])['item_cnt_day']
    item_sales_grp = monthly_item_sales_df.groupby(['item_id'])['item_cnt_day']
    output = {}
    for groupby_obj, name in [
        (category_sales_grp, 'category'),
        (shop_sales_grp, 'shop'),
        (item_sales_grp, 'item'),
    ]:
        fmt = 'Overall_{}'.format(name) + '_{}'
        arr = []
        arr += [groupby_obj.mean().to_frame(fmt.format('mean'))]
        arr += [groupby_obj.std().to_frame(fmt.format('std')).fillna(0)]
        arr += [groupby_obj.min().to_frame(fmt.format('min'))]
        arr += [groupby_obj.max().to_frame(fmt.format('max'))]
        arr += [groupby_obj.quantile(0.1).to_frame(fmt.format('qt_' + str(10)))]
        arr += [groupby_obj.quantile(0.25).to_frame(fmt.format('qt_' + str(25)))]
        arr += [groupby_obj.quantile(0.5).to_frame(fmt.format('qt_' + str(50)))]
        arr += [groupby_obj.quantile(0.75).to_frame(fmt.format('qt_' + str(75)))]
        arr += [groupby_obj.quantile(0.95).to_frame(fmt.format('qt_' + str(95)))]

        df = pd.concat(arr, axis=1)
        output[name] = df

    return output


class OverallFeatures:
    def __init__(self, sales_df, items_df):
        self._sales_df = sales_df
        self._items_df = items_df
        self._overall_features_dict = overall_features(sales_df, items_df)

    def item_features(self, item_id):
        return self._overall_features_dict['item'].loc[item_id]

    def shop_features(self, shop_id):
        return self._overall_features_dict['shop'].loc[shop_id]

    def category_features(self, category_id):
        return self._overall_features_dict['category'].loc[category_id]
