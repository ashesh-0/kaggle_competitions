"""
Couple of functions needed for training the model.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split


def eval_metric(actual, predic, molecule_type):
    error = np.abs(actual - predic)
    df = pd.DataFrame(np.vstack([error, molecule_type]).T, columns=['error', 'type'])
    df['error'] = df['error'].astype(np.float32)
    return df.groupby('type')['error'].mean().apply(np.log).mean()


def get_performance(model, X_train_df, y_train_df, X_test_df, y_test_df, raw_train_df):
    molecule_type_train = raw_train_df.loc[X_train_df.index, 'type']
    molecule_type_test = raw_train_df.loc[X_test_df.index, 'type']

    prediction_train = model.predict(X_train_df)
    prediction_test = model.predict(X_test_df)

    train_perf = eval_metric(y_train_df.values, prediction_train, molecule_type_train)
    test_perf = eval_metric(y_test_df.values, prediction_test, molecule_type_test)
    return (train_perf, test_perf)


def train(catboost_config, train_X, train_Y, test_X, test_Y, data_size, plot=False):

    train_pool = Pool(train_X, label=train_Y)
    if test_X is None:
        test_pool = None
        test_X = []
    else:
        test_pool = Pool(test_X, label=test_Y)
    print('{:.2f}% Train:{}K Test{}K'.format(100 * (len(train_X) + len(test_X)) / data_size,
                                             len(train_X) // 1000,
                                             len(test_X) // 1000))
    model = CatBoostRegressor(**catboost_config)
    model.fit(train_pool, eval_set=test_pool, plot=plot, logging_level='Silent')
    return model


def train_for_each_output(catboost_config_for_each_output, train_X_df, Y_df, no_validation=False):
    models = {}
    predictions = []
    if no_validation:
        train_X = train_X_df
        train_Y = Y_df
        test_X = None
        test_Y = None
    else:
        train_X, test_X = train_test_split(train_X_df, test_size=0.15, random_state=0)
        test_Y = Y_df.loc[test_X.index]
        train_Y = Y_df.loc[train_X.index]

    for output_type in Y_df.columns:
        print(output_type)
        models[output_type] = train(
            catboost_config_for_each_output[output_type],
            train_X,
            train_Y[output_type],
            test_X,
            test_Y[output_type],
            train_X_df.shape[0],
        )
        print('Best iter', models[output_type].best_iteration_, catboost_config_for_each_output[output_type])

        if no_validation is False:
            predictions.append(pd.Series(models[output_type].predict(test_X), index=test_X.index))
            print('Eval',
                  eval_metric(test_Y[output_type].values, predictions[-1].values, X.loc[test_Y.index, 'type_enc']))

    if no_validation is False:
        prediction_df = pd.concat(predictions, axis=1).sum(axis=1)
        actual = Y_df.loc[prediction_df.index].sum(axis=1)
        print('Final metric', eval_metric(actual.values, prediction_df.values, X.loc[actual.index, 'type_enc']))
        print('')
        single_model = train(
            catboost_config_for_each_output['single'],
            train_X,
            train_Y.sum(axis=1),
            test_X,
            test_Y.sum(axis=1),
            train_X_df.shape[0],
        )
        print('Eval on single model',
              eval_metric(test_Y.sum(axis=1).values, single_model.predict(test_X), X.loc[test_Y.index, 'type_enc']))

    return models


def train_for_each_type(catboost_config_for_each_type, train_X_df, Y_df, no_validation=False):
    models = {}
    anal_dict = {}
    predictions = []
    for type_enc in train_X_df['type_enc'].unique():
        print(type_enc)
        anal_dict[type_enc] = {}
        X_t = train_X_df[train_X_df.type_enc == type_enc]

        if no_validation:
            train_X = X_t
            test_X = None
            test_Y = None
        else:
            train_X, test_X = train_test_split(X_t, test_size=0.15, random_state=0)
            test_Y = Y_df.loc[test_X.index]

        train_Y = Y_df.loc[train_X.index]
        models[type_enc] = train(catboost_config_for_each_type[type_enc], train_X, train_Y, test_X, test_Y,
                                 train_X_df.shape[0])
        anal_dict[type_enc]['train'] = pd.Series(models[type_enc].predict(train_X), index=train_X.index)
        print('Best iter', models[type_enc].best_iteration_, catboost_config_for_each_type[type_enc])

        if no_validation is False:
            anal_dict[type_enc]['test'] = pd.Series(models[type_enc].predict(test_X), index=test_X.index)
            predictions.append(anal_dict[type_enc]['test'])
            print('Eval', eval_metric(test_Y.values, predictions[-1].values, test_X['type_enc']))

    if no_validation is False:
        prediction_df = pd.concat(predictions)
        actual = Y_df.loc[prediction_df.index]
        print('Final metric', eval_metric(actual.values, prediction_df.values, X.loc[actual.index, 'type_enc']))

    return (models, anal_dict)
