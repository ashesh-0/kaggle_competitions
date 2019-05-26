"""
Classifies whether next month sales are 0/20
"""
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt


def plot_importance(model):
    importances = model.get_feature_importance(prettified=True)
    x = np.array(list(range(len(importances))))
    y = [imp for (_, imp) in importances]
    my_xticks = [col for (col, _) in importances]
    fig, ax = plt.subplots(figsize=(10, 20))
    width = 0.75  # the width of the bars
    ax.barh(x, y, width, color="blue")
    ax.set_yticks(x + width / 2)
    ax.set_yticklabels(my_xticks, minor=False)
    plt.title('Feature Importances')
    plt.xlabel('importance')


def get_model(X_train_df, y_train_df, model_kwargs, X_val_df=None, y_val_df=None):
    hs_model = CatBoostClassifier(**model_kwargs)
    if X_val_df is None:
        eval_set = None
    else:
        eval_set = (X_val_df, y_val_df)
    hs_model.fit(
        X_train_df,
        y_train_df,
        eval_set=eval_set,
        verbose=1000,
    )
    return hs_model


def descritize_y(y_df, low_threshold=1, high_threshold=19):
    y_df[y_df < low_threshold] = 0
    y_df[(y_df > 0) & (y_df < high_threshold)] = 1
    y_df[y_df >= high_threshold] = 2


def high_sales_y(discritized_y):
    max_val = discritized_y.max()
    discritized_y[discritized_y < max_val] = 0
    discritized_y[discritized_y > 0] = 1


def low_sales_y(discritized_y):
    min_val = discritized_y.min()
    assert min_val == 0
    discritized_y[discritized_y > 0] = 1


class ExtremeSalesClassifier:
    def __init__(
            self,
            sales_df: pd.DataFrame,
            X_df: pd.DataFrame,
            y_df: pd.Series,
            high_model_kwargs: dict,
            low_model_kwargs: dict,
    ):
        self._sales_df = sales_df
        self._X = X_df

        self._y = y_df.copy()
        descritize_y(self._y)

        self._high_feature = None
        self._low_feature = None
        self._last_train_dbn = 33
        self.feature_df = None
        overfitted_columns = ['item_id', 'orig_item_id', 'date_block_num']
        self._train_cols = X_df.columns[~X_df.columns.isin(overfitted_columns)].tolist()
        self._high_kwargs = high_model_kwargs
        self._low_kwargs = low_model_kwargs

    def get_features_for_test(self, test_df):
        X_tr = self.get_X(self._X)
        X_test = self.get_X(test_df)

        y_tr_high = self._y.copy()
        high_sales_y(y_tr_high)
        model_high = get_model(X_tr, y_tr_high, self._high_kwargs)
        high_feature = pd.Series(model_high.predict_proba(X_test)[:, 1], index=X_test.index)
        plot_importance(model_high)

        y_tr_low = self._y.copy()
        low_sales_y(y_tr_low)
        model_low = get_model(X_tr, y_tr_low, self._low_kwargs)
        low_feature = pd.Series(model_low.predict_proba(X_test)[:, 1], index=X_test.index)
        plot_importance(model_low)
        return pd.concat([high_feature.to_frame('high_sales'), low_feature.to_frame('low_sales')], axis=1)

    def get_X(self, df):
        return df[self._train_cols]

    def run(self, starting_dbn: int = 0):
        index = np.zeros(self._X.shape[0])
        high_sales_feature = np.zeros(index.shape[0])
        low_sales_feature = np.zeros(index.shape[0])
        index_idx = 0
        for train_end_dbn in tqdm_notebook(range(starting_dbn, self._last_train_dbn)):
            X_tr = self.get_X(self._X[self._X.date_block_num <= train_end_dbn])
            X_val = self.get_X(self._X[self._X.date_block_num == 1 + train_end_dbn])
            X_test = self.get_X(self._X[self._X.date_block_num == 2 + train_end_dbn])

            if index_idx == 0:
                index_idx = X_tr.shape[0] + X_val.shape[0]

            index[index_idx:index_idx + X_test.shape[0]] = X_test.index.values

            y_tr_high = self._y.loc[X_tr.index].copy()
            y_val_high = self._y.loc[X_val.index].copy()

            high_sales_y(y_tr_high)
            high_sales_y(y_val_high)
            print('Training for date_block_num', train_end_dbn)
            model_high = get_model(X_tr, y_tr_high, self._high_kwargs, X_val, y_val_high)

            high_sales_feature[index_idx:index_idx + X_test.shape[0]] = model_high.predict_proba(X_test)[:, 1]

            del y_tr_high, y_val_high, model_high

            y_tr_low = self._y.loc[X_tr.index].copy()
            y_val_low = self._y.loc[X_val.index].copy()

            low_sales_y(y_tr_low)
            low_sales_y(y_val_low)

            model_low = get_model(X_tr, y_tr_low, self._low_kwargs, X_val, y_val_low)

            low_sales_feature[index_idx:index_idx + X_test.shape[0]] = model_low.predict_proba(X_test)[:, 1]
            index_idx = index_idx + X_test.shape[0]
            del y_tr_low, y_val_low, model_low

        self.feature_df = pd.DataFrame(
            np.vstack([high_sales_feature, low_sales_feature]).T, index=index, columns=['high_sales', 'low_sales'])
