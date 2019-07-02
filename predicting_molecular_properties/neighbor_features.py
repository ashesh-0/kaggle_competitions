"""
From kaggle discussions, I see that neighbour information is quite helpful. From neighbor, I mean if there is a bond
between atom_index 4 and 5 then for 4 neighbors will be 1,2,3 and for 5, neighbors will be 6,7,8.
Note that in a generic case, there can be low_neighbors (with a lower index) or high neighbors (with a higher index)
"""
import numpy as np
import pandas as pd
# from common_utils import find_distance_btw_point, find_cos
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder


def add_neighbors_features(df, structures_df, atom_encoder, neighbor_cnt=3):
    """
    # 1. Distance from neighbor.
    # 2. Atom type of neighbor.
    # 3. Stats over these.
    """
    df.reset_index(inplace=True)

    assert 'atom_index' not in df.columns

    atom_present = False
    if 'atom' in df.columns:
        atom_present = True
    else:
        df['atom'] = 0

    sfx_fmt = '_{}neighbor_atom_index_{}'
    for a_i in [0, 1]:
        angle_cols = []
        dist_cols = []
        atom_cols = []
        for idx in tqdm_notebook(range(-neighbor_cnt, neighbor_cnt + 1, 1)):
            if idx == 0:
                continue

            df['neighbor_idx'] = df[f'atom_index_{a_i}'] + idx

            sfx = sfx_fmt.format(idx, a_i)
            # atom type will get added
            df = pd.merge(
                df,
                structures_df,
                left_on=['molecule_name', 'neighbor_idx'],
                right_on=['molecule_name', 'atom_index'],
                suffixes=('', sfx),
                how='left')

            angle_col = 'angle' + sfx
            angle_cols.append(angle_col)

            df[angle_col] = find_cos(df, 'x', 'y', 'z', f'x_{a_i}', f'y_{a_i}', f'z_{a_i}', f'x_{1-a_i}', f'y_{1-a_i}',
                                     f'z_{1-a_i}')
            df[angle_col] = df[angle_col].astype(np.float32)

            dis_col = 'dis' + sfx
            dist_cols.append(dis_col)

            df[dis_col] = find_distance_btw_point(df, f'x_{a_i}', f'y_{a_i}', f'z_{a_i}', 'x', 'y', 'z')
            df[dis_col] = df[dis_col].astype(np.float32)

            atom_col = 'atom' + sfx
            atom_cols.append(atom_col)
            df[atom_col] = df[atom_col].fillna('NAN')
            df.drop(['x', 'y', 'z', 'atom_index'], axis=1, inplace=True)

        # angle stats
        df['avg_angle' + sfx_fmt.format('all', a_i)] = np.nanmean(df[angle_cols], axis=1)
        # fillna after stats computation done.
        df[angle_cols] = df[angle_cols].fillna(100)

        # dist nan
        df['avg_dis' + sfx_fmt.format('all', a_i)] = np.nanmean(df[dist_cols], axis=1)
        df[dist_cols] = df[dist_cols].fillna(100)

        # cnt stats
        df[f'H_cnt_neighbors_atom_index_{a_i}'] = (df[atom_cols] == 'H').sum(axis=1)
        df[f'C_cnt_neighbors_atom_index_{a_i}'] = (df[atom_cols] == 'C').sum(axis=1)
        for c in atom_cols:
            df[c] = atom_encoder.transform(df[c]).astype(np.uint8)

    df.drop('neighbor_idx', axis=1, inplace=True)
    if not atom_present:
        df.drop('atom', axis=1, inplace=True)

    df.set_index('id', inplace=True)
    return df
