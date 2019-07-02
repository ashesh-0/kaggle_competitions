import pandas as pd
from common_utils import find_cos


def test_find_cos():
    df = pd.DataFrame([[0, 0, 0, 1, 0, 0, 1, 1, 0]])
    df.loc[1] = [0, 0, 0, 1, 2, 0, -1, 3, 0]
    df.loc[2] = [0, 0, 0, 1, 2, 0, 2, 4, 0]
    cos = find_cos(df, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    assert cos.loc[0] == 0
    assert cos.loc[1] == 0
    assert abs(cos.loc[2] + 1) < 1e-6
