import numpy as np
import pandas as pd


class IdFeatures:
    """
    We will change the id of item and shop so that it becomes easier for trees to split data.
    """

    def __init__(self, sales_df: pd.DataFrame):
        self._sales_df = sales_df

        self._item_id_old_to_new = None
        self._shop_id_old_to_new = None

        self.fit()

    def _fit(self, id_name):
        monthly_df = self._sales_df.groupby(['date_block_num', id_name])['item_cnt_day'].sum().unstack().fillna(0)
        non_zero_months = (monthly_df > 0).sum()

        avg_monthly_sales_df = (monthly_df.sum() / non_zero_months).to_frame('avg_sales')
        avg_monthly_sales_df = avg_monthly_sales_df.sort_values('avg_sales')
        avg_monthly_sales_df['new_id'] = list(range(avg_monthly_sales_df.shape[0]))
        return avg_monthly_sales_df['new_id'].to_dict()

    def fit(self):
        self._item_id_old_to_new = self._fit('item_id')
        self._shop_id_old_to_new = self._fit('shop_id')

    def get_features(self, item_id, shop_id):
        return np.array([[self._item_id_old_to_new[item_id], self._shop_id_old_to_new[shop_id]]])


if __name__ == '__main__':
    cols = ['date_block_num', 'item_id', 'shop_id', 'item_cnt_day']
    df = pd.DataFrame(
        [
            [0, 0, 0, 5],
            [0, 0, 0, 10],
            [1, 0, 0, 4],
            [1, 0, 0, 6],
            [0, 1, 0, 1],
            [0, 1, 0, 2],
            [1, 1, 0, 4],
            [1, 1, 0, 6],
            [0, 2, 1, 10],
            [0, 2, 1, 2],
            [1, 2, 1, 4],
            [1, 2, 1, 6],
        ],
        columns=cols)

    f = IdFeatures(df)
    print(f.get_features(0, 0))
