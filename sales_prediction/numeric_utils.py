from multiprocessing import Pool
import pandas as pd


def run(fn_args):
    fn, args, name = fn_args
    print('Starting', args)
    return fn(*args).to_frame(name)


def compute_concurrently(args, process_count=4):

    with Pool(processes=process_count) as pool:
        output = pool.map(run, args)

    df = pd.concat(output, axis=1)
    return df.reset_index(level=0).drop('shop_item_group', axis=1)
