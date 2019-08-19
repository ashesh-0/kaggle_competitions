import numpy as np


class Scaler:
    """
    Changes inplace.
    Transforms columns one by one. This makes it memory efficient.
    Computes mean and std ignoring the nans.
    """

    def __init__(self, skip_columns=None, remove_mean=True, divide_by_std=True, fillna_with_mean=True):
        self._skip_columns = [] if skip_columns is None else skip_columns
        self._remove_mean = remove_mean
        self._divide_by_std = divide_by_std
        self._fillna_with_mean = fillna_with_mean
        self._mean_dict = {}
        self._std_dict = {}
        self._is_fitted = False

    def fit(self, df):
        for col in df.columns:
            if col in self._skip_columns:
                continue
            self._mean_dict[col] = np.nanmean(df[col].values)
            self._std_dict[col] = np.sqrt(np.nanvar(df[col].values))

        self._is_fitted = True

    def transform(self, df):
        assert self._is_fitted is True, 'Call fit() first'

        for col in df.columns:
            if col in self._skip_columns:
                continue

            if self._remove_mean:
                if col in self._mean_dict:
                    df[col] = df[col] - self._mean_dict[col]
                else:
                    print(f'Unexpectedly no mean present for "{col}"!! Skipping it')

            if self._divide_by_std:
                if col in self._std_dict:
                    df[col] = df[col] / self._std_dict[col]
                else:
                    print(f'Unexpectedly no std present for "{col}"!! Skipping it')

        if self._fillna_with_mean:
            df.fillna(0, inplace=True)

    def fit_transform(self, df):
        self.fit(df)
        self.transform(df)
