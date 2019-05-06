from multiprocessing import Pool
import pandas as pd


def run(fn_args):
    fn, args, kwargs = fn_args
    return fn(*args, **kwargs)


def compute_concurrently(args, process_count=4):

    with Pool(processes=process_count) as pool:
        output = pool.map(run, args)

    df = pd.concat(output, axis=1)
    return df


def get_items_in_market(sales_df, block_num):
    """
    Returns list of item_ids which are traded from the first day of given month
    """
    last_trading_month = sales_df.groupby(['item_id'])['date_block_num'].max()
    first_trading_month = sales_df.groupby(['item_id'])['date_block_num'].min()
    it1 = set(last_trading_month[last_trading_month >= block_num].index.tolist())
    it2 = set(first_trading_month[first_trading_month <= block_num].index.tolist())
    output = list(it1.intersection(it2))
    output.sort()
    return output


def get_shops_in_market(sales_df, block_num):
    """
    Returns list of shop_ids which are open in that month.
    """
    last_trading_month = sales_df.groupby(['shop_id'])['date_block_num'].max()
    first_trading_month = sales_df.groupby(['shop_id'])['date_block_num'].min()
    it1 = set(last_trading_month[last_trading_month >= block_num].index.tolist())
    it2 = set(first_trading_month[first_trading_month <= block_num].index.tolist())
    output = list(it1.intersection(it2))
    output.sort()
    return output
