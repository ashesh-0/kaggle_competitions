import itertools
import pandas as pd
import numpy as np
from model_validator import ModelValidator


def test_model_validator_should_make_last_month_as_validation():
    sales_df = pd.DataFrame(
        [
            [32, 1, 1],
            [32, 1, 1],
            [32, 1, 2],
            [32, 2, 1],
            [32, 2, 2],
            [32, 2, 2],
            [32, 3, 2],
            # now the validation set.
            [33, 2, 1],
            [33, 2, 1],
            [33, 1, 1],
        ],
        columns=['date_block_num', 'item_id', 'shop_id'])
    sales_df.index += 10

    # index = sales_df.index.tolist()
    # np.random.shuffle(index)

    X = pd.DataFrame(np.random.rand(sales_df.shape[0], 10), index=sales_df.index)
    y = pd.Series(np.random.rand((X.shape[0])), index=X.index)

    expected_train_X = X.loc[sales_df.iloc[:-3].index]
    expected_train_y = y.loc[expected_train_X.index]
    mv = ModelValidator(sales_df, X, y)
    train_X, train_y, val_X, val_y = mv.get_train_val_data()

    assert expected_train_X.equals(train_X)
    assert expected_train_y.equals(train_y)

    expected_nz_val_X = X.iloc[[-3, -1]]
    expected_nz_val_y = y[expected_nz_val_X.index]
    nz_val_y = val_y[val_y != 0]
    nz_val_X = X.loc[nz_val_y.index]

    assert nz_val_X.equals(expected_nz_val_X)
    assert nz_val_y.equals(expected_nz_val_y)
