import gc
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from constants import DATA_FPATH, TEST_LIKE_SALES_FPATH, SALES_FPATH, ITEMS_FPATH, SHOPS_FPATH, TEST_SALES_FPATH

orig_sales_df = pd.read_csv(SALES_FPATH)
items_df = pd.read_csv(ITEMS_FPATH)
shops_df = pd.read_csv(SHOPS_FPATH)
test_sales_df = pd.read_csv(TEST_SALES_FPATH, index_col=0)

# Load preprocessed data.
X_df = pd.read_hdf(DATA_FPATH, 'X')
y_df = pd.read_hdf(DATA_FPATH, 'y')
val_X_10df = pd.read_hdf(DATA_FPATH, 'val_X_10')
val_y_10df = pd.read_hdf(DATA_FPATH, 'val_y_10')
val_X_9df = pd.read_hdf(DATA_FPATH, 'val_X_9')
val_y_9df = pd.read_hdf(DATA_FPATH, 'val_y_9')
columns = X_df.columns.tolist()

sales_df = pd.read_hdf(TEST_LIKE_SALES_FPATH, 'df')

val_X_df = pd.concat([val_X_9df, val_X_10df], sort=True)
val_y_df = pd.concat([val_y_9df, val_y_10df])
assert val_X_df.index.equals(val_y_df.index)
del val_X_9df, val_X_10df
del val_y_9df, val_y_10df

# trim y to range [0,20]
y_df[y_df > 20] = 20
y_df[y_df < 0] = 0

val_y_df[val_y_df > 20] = 20
val_y_df[val_y_df < 0] = 0

# get data.
validation_date_blocks = val_X_df.join(sales_df[[]], how='inner')['date_block_num'].unique().tolist()
print(validation_date_blocks)
min_val_dbn = min(validation_date_blocks)
X_train_df = X_df.loc[sales_df[sales_df.date_block_num < min_val_dbn].index]
y_train_df = y_df.loc[X_train_df.index]

del X_df, y_df

float64_cols = X_train_df.dtypes[X_train_df.dtypes == np.float64].index.tolist()
X_train_df[float64_cols] = X_train_df[float64_cols].astype(np.float32)
val_X_df[float64_cols] = val_X_df[float64_cols].astype(np.float32)

overfitted_cols = ['item_id', 'orig_item_id', 'date_block_num']

X_train_df.drop(overfitted_cols, axis=1, inplace=True)
val_X_df.drop(overfitted_cols, axis=1, inplace=True)

assert set(X_train_df.columns) == set(val_X_df.columns)
val_X_df = val_X_df[X_train_df.columns]
assert val_X_df.columns.equals(X_train_df.columns)

del sales_df
gc.collect()

from catboost import Pool
pool_train = Pool(data=X_train_df, label=y_train_df)
pool_val = Pool(data=val_X_df, label=val_y_df)

from catboost import CatBoostRegressor
regressor_kwargs = {
    'depth': 11,
    'iterations': 5000,
    'random_seed': 0,
    'learning_rate': 0.02,
    #     'custom_metric':'R2',
    'od_type': 'Iter',
    'od_wait': 300,
    'task_type': 'GPU',
    #     'train_dir':'depth_8',
}
model = CatBoostRegressor(**regressor_kwargs)

model.fit(pool_train, eval_set=pool_val, verbose=100, plot=True)
