import pandas as pd
import numpy as np
from scipy import stats


def get_outliers(df, outlier_z_score_abs_threshold=5, outlier_feature_fraction=0.3):
    # features is level 1. ts is level 0.
    f1 = df.groupby(df.columns.get_level_values(1), axis=1).mean()
    f2 = df.abs().groupby(df.columns.get_level_values(1), axis=1).mean()
    f3 = df.groupby(df.columns.get_level_values(1), axis=1).std()
    f4 = df.groupby(df.columns.get_level_values(1), axis=1).quantile(0.5)
    f5 = df.groupby(df.columns.get_level_values(1), axis=1).quantile(0.1)
    f6 = df.groupby(df.columns.get_level_values(1), axis=1).quantile(0.9)

    f1 = f1.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)
    f2 = f2.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)
    f3 = f3.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)
    f4 = f4.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)
    f5 = f5.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)
    f6 = f6.drop(['diff_smoothend_by_1 Quant-0.0'], axis=0)

    outlier_feature_count = int(f1.shape[0] * outlier_feature_fraction)
    fs = [f1, f2, f3, f4, f5, f6]
    zscores = list(map(lambda f: stats.zscore(f, axis=1), fs))
    outliers = list(map(lambda zscore: np.abs(zscore) > outlier_z_score_abs_threshold, zscores))
    examplewise_outliers = list(map(lambda outlier: np.sum(outlier, axis=0), outliers))
    # print('Shape of example_outliers', examplewise_outliers[0].shape)
    outlier_filters = []
    for i, ex_out in enumerate(examplewise_outliers):
        outlier_filter = ex_out > outlier_feature_count
        outlier_filters.append(outlier_filter)
        # zero_percent = round(ex_out[ex_out == 0].shape[0] / ex_out.shape[0] * 100, 2)
        # outlier_count = ex_out[outlier_filter].shape[0]
        # print('FeatureIndex', i, 'zero percent', zero_percent)
        # print('FeatureIndex', i, 'outlier count', outlier_count)

    outlier_filter = np.sum(outlier_filters, axis=0)
    outlier_filter = outlier_filter >= len(outlier_filters) // 2
    print('Outlier percent:', round(outlier_filter.sum() / outlier_filter.shape[0] * 100, 2))
    print('Outlier count:', outlier_filter.sum())

    output_df = pd.Series(outlier_filter, index=f1.columns).to_frame('outliers')
    output_df.index = output_df.index.astype(int)
    return output_df


def target_class_outlier_distribution_grid_search(df, meta_fname):
    meta_df = pd.read_csv(meta_fname).set_index('signal_id')[['target']]
    thresholds = [4, 5, 6]
    fractions = [0.3, 0.5, 0.7]
    vc = round(meta_df['target'].value_counts().loc[1] / meta_df.shape[0] * 100, 2)
    print('In original data, target class 1 is {}%'.format(vc))

    target_one_percent = pd.DataFrame([], columns=thresholds, index=fractions)
    target_one_percent.index.name = 'outlier_feature_fraction'
    target_one_percent.columns.name = 'outlier_z_score_abs_threshold'
    for thresh in thresholds:
        for frac in fractions:
            outliers_df = get_outliers(df, thresh, frac)
            outliers_df = outliers_df.join(meta_df, how='left')
            vc = outliers_df[outliers_df['outliers'] == True]['target'].value_counts()
            if 1 not in vc.index:
                target_one_percent.loc[frac, thresh] = 0
            else:
                percnt = round(vc.loc[1] / vc.sum() * 100, 2)
                target_one_percent.loc[frac, thresh] = percnt
                print('In outliers with thresh:{}, frac:{}, target class 1 is {}%'.format(thresh, frac, percnt))

    return target_one_percent
