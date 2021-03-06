import numpy as np
import pandas as pd
from sortedcontainers import SortedList


def nmonths_features(sales_df, col_name, num_months, quantiles):
    """
    It does not use that month's data to compute features. it uses previous months data.
    Args:
        sales_df: it  is sorted by shop_id, item_id and date_block_num.
            Also, shop_item_group value changes when either the shop or the item changes.
    """
    assert num_months >= 1
    columns = ['sum', 'min', 'max'] + ['{}_q'.format(q) for q in quantiles]
    columns = ['{}_{}M_{}'.format(col_name, num_months, c) for c in columns]

    # For fast processing
    values = sales_df[col_name].values
    indices = sales_df.index.tolist()

    output_data = np.zeros((len(indices), len(columns)))
    s_list = SortedList()

    rolling_sum = 0
    tail_index_position = 0
    head_index_position = -1
    tail_index = indices[tail_index_position]
    tail_dbn = sales_df.iloc[0]['date_block_num']

    cur_group = sales_df.iloc[0]['shop_item_group']
    for head_index in indices:
        head_group = sales_df.at[head_index, 'shop_item_group']
        head_index_position += 1
        head_dbn = sales_df.at[head_index, 'date_block_num']

        if head_group != cur_group:
            s_list = SortedList()
            cur_group = head_group
            tail_index = head_index
            tail_index_position = head_index_position
            tail_dbn = sales_df.at[head_index, 'date_block_num']
            rolling_sum = 0
        else:
            if head_index_position >= 1:
                item = values[head_index_position - 1]
                s_list.add(item)
                rolling_sum += item

        while head_dbn - tail_dbn > num_months:

            value_to_be_removed = values[tail_index_position]
            s_list.remove(value_to_be_removed)
            rolling_sum -= value_to_be_removed

            tail_index_position += 1
            tail_index = indices[tail_index_position]
            tail_dbn = sales_df.at[tail_index, 'date_block_num']

        if len(s_list) == 0:
            output_data[head_index_position, :] = [0] * output_data.shape[1]
        else:
            # compute values on data starting at tail_index_position and ending at head_index_position, both inclusive.
            quants = [s_list[max(0, int(len(s_list) * q) - 1)] for q in quantiles]
            output_data[head_index_position, :] = [rolling_sum, s_list[0], s_list[-1]] + quants

    return pd.DataFrame(output_data, columns=columns, index=sales_df.index)
