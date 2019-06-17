import pandas as pd


def get_unexpected_category_entries_filter(val_df, sales_df, items_df):
    min_dbn = val_df.date_block_num.min()
    sales_df = sales_df[sales_df.date_block_num < min_dbn]
    sales_df = sales_df[sales_df.item_cnt_day > 0]
    if 'item_category_id' not in sales_df:
        sales_df = pd.merge(sales_df, items_df[['item_id', 'item_category_id']], how='left', on='item_id')

    assert 'item_category_id' in val_df

    sales_df['shop_category_id'] = sales_df['shop_id'] * 100 + sales_df['item_category_id']

    assert 'shop_category_id' not in val_df
    val_df['shop_category_id'] = val_df['shop_id'] * 100 + val_df['item_category_id']

    new_item_list = set(val_df.item_id.values) - set(sales_df.item_id.values)
    new_shop_list = set(val_df.shop_id.values) - set(sales_df.shop_id.values)

    old_items_filter = ~val_df.item_id.isin(new_item_list)
    old_shops_filter = ~val_df.shop_id.isin(new_shop_list)
    old_item_shops_filter = old_items_filter & old_shops_filter

    new_shop_category_id_list = set(val_df.shop_category_id.values) - set(sales_df.shop_category_id.values)

    zero_filter = val_df.shop_category_id.isin(new_shop_category_id_list) & old_item_shops_filter

    val_df.drop('shop_category_id', inplace=True, axis=1)
    return zero_filter
