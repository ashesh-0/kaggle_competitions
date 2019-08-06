"""
Couple of functions needed for training the model.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from tqdm import tqdm_notebook


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


def aversarial_train(train_X, train_Y, test_X, test_Y, plot=False):
    """
    Here, we try to differentiate between train and test
    """
    train_pool = Pool(train_X, label=train_Y)
    test_pool = Pool(test_X, label=test_Y)

    model = CatBoostClassifier(
        iterations=500,
        random_seed=63,
        learning_rate=0.5,
        custom_loss=['AUC'],
        task_type='GPU',
        depth=10,
    )
    model.fit(train_pool, eval_set=test_pool, plot=plot, logging_level='Silent')
    auc_train = model.best_score_['learn']['AUC']
    auc_test = model.best_score_['validation']['AUC']
    return (model, auc_train, auc_test)


def train_lightGBM(lightGBM_config, train_X, train_Y, test_X, test_Y, data_size):
    import lightgbm as lgb

    lgb_train = lgb.Dataset(train_X, train_Y)
    valid_sets = [lgb_train]

    # When creating final model, test_X is not None but test_Y is None.
    if test_Y is None:
        lgb_eval = None
        test_Y = []
    else:
        lgb_eval = lgb.Dataset(test_X, test_Y, reference=lgb_train)
        valid_sets.append(lgb_eval)

    print('{:.2f}% Train:{}K Test:{}K #Features:{}'.format(
        100 * (len(train_Y) + len(test_Y)) / data_size,
        len(train_Y) // 1000,
        len(test_Y) // 1000,
        train_X.shape[1],
    ))
    model = lgb.train(
        lightGBM_config,
        lgb_train,
        valid_sets=valid_sets,
        verbose_eval=5000,
        early_stopping_rounds=40,
    )

    feature_importance_df = pd.DataFrame(
        sorted(zip(model.feature_importance(), test_X.columns)), columns=['Importances',
                                                                          'Feature Index']).set_index('Feature Index')
    return (model, feature_importance_df, model.best_iteration)


def train_catboost(catboost_config, train_X, train_Y, test_X, test_Y, data_size, plot=False):

    train_pool = Pool(train_X, label=train_Y)
    # When creating final model, test_X is not None but test_Y is None.
    if test_Y is None:
        test_pool = None
        test_Y = []
    else:
        test_pool = Pool(test_X, label=test_Y)
    print('{:.2f}% Train:{}K Test:{}K #Features:{}'.format(
        100 * (len(train_Y) + len(test_Y)) / data_size,
        len(train_Y) // 1000,
        len(test_Y) // 1000,
        train_X.shape[1],
    ))
    model = CatBoostRegressor(**catboost_config)
    model.fit(train_pool, eval_set=test_pool, plot=plot, logging_level='Silent')
    feature_importance_df = model.get_feature_importance(prettified=True).set_index('Feature Index')
    return (model, feature_importance_df, model.best_iteration_)


def averserial_train_for_each_type_no_model(useless_cols_for_each_type, train_X_df, test_X_df, max_auc):
    """
    It does not store (and return) model. It is very lean in terms of memory in that sense.
    We try to see how much AUC do we get when we try to differentiate train with test.
    """
    anal_dict = {}
    for type_enc in train_X_df['type_enc'].unique():

        anal_dict[type_enc] = {'train': None, 'test': None, 'feature_importance': None, 'useless_cols': []}
        X_t = train_X_df[train_X_df.type_enc == type_enc].copy()
        test_X_t = test_X_df[test_X_df.type_enc == type_enc].copy()

        if len(useless_cols_for_each_type[type_enc]) > 0:
            X_t = X_t.drop(useless_cols_for_each_type[type_enc], axis=1)
            test_X_t = test_X_t[X_t.columns]

        X = pd.concat([X_t, test_X_t], axis=0)
        Y = X[[]].copy()
        Y['target'] = 1
        Y.loc[test_X_t.index, 'target'] = 0
        train_X, test_X = train_test_split(X, test_size=0.15, random_state=0, stratify=Y['target'])
        test_Y = Y.loc[test_X.index]
        train_Y = Y.loc[train_X.index]

        (model, auc_train, auc_test) = aversarial_train(train_X, train_Y, test_X, test_Y)
        anal_dict[type_enc]['train'] = round(auc_train, 2)
        anal_dict[type_enc]['test'] = round(auc_test, 2)
        f_df = model.get_feature_importance(prettified=True).set_index('Feature Index')['Importances']
        # anal_dict[type_enc]['feature_importance'] = f_df.to_dict()
        while auc_test > max_auc:
            useless_cols = f_df[f_df > 10].index.tolist()
            if len(useless_cols) == 0:
                break
            train_X.drop(useless_cols, axis=1, inplace=True)
            test_X.drop(useless_cols, axis=1, inplace=True)
            print('Typ:', type_enc, 'Train:', anal_dict[type_enc]['train'], 'Test:', anal_dict[type_enc]['test'],
                  'Removing:', useless_cols)
            anal_dict[type_enc]['useless_cols'] += useless_cols

            (model, auc_train, auc_test) = aversarial_train(train_X, train_Y, test_X, test_Y)
            anal_dict[type_enc]['train'] = round(auc_train, 2)
            anal_dict[type_enc]['test'] = round(auc_test, 2)
            f_df = model.get_feature_importance(prettified=True).set_index('Feature Index')['Importances']
            # anal_dict[type_enc]['feature_importance'] = f_df.to_dict()

    return anal_dict


def one_type_eval_metric(actual, predic):
    """
    Metric for just one type
    """
    return eval_metric(actual, predic, np.zeros(actual.shape))


def permutation_importance(model, X_val, y_val, metric, threshold=0.005, verbose=True):
    """
    Permutes the features. If performance doesn't change a lot then it is useless.
    """
    # Taken from here https://www.kaggle.com/speedwagon/permutation-importance
    results = {}

    y_pred = model.predict(X_val)

    results['base_score'] = metric(y_val, y_pred)
    if verbose:
        print(f'Base score {results["base_score"]:.5}')

    for col in tqdm_notebook(X_val.columns):
        freezed_col = X_val[col].copy()

        X_val[col] = np.random.permutation(X_val[col])
        preds = model.predict(X_val)
        results[col] = metric(y_val, preds)

        X_val[col] = freezed_col

        if verbose:
            print(f'column: {col} - {results[col]:.5}')

    bad_features = [k for k in results if results[k] > results['base_score'] + threshold]

    return results, bad_features


def permute_to_get_useless_features(catboost_config_for_each_type, useless_cols_for_each_type, train_X_df, Y_df):
    bad_features_dict = {}
    for type_enc in train_X_df['type_enc'].unique():
        print(type_enc)
        bad_features_dict[type_enc] = {}
        X_t = train_X_df[train_X_df.type_enc == type_enc].copy()

        if len(useless_cols_for_each_type[type_enc]) > 0:
            X_t = X_t.drop(useless_cols_for_each_type[type_enc], axis=1)

        train_X, test_X = train_test_split(X_t, test_size=0.15, random_state=0)
        test_Y = Y_df.loc[test_X.index].copy()

        train_Y = Y_df.loc[train_X.index].copy()
        model, _ = train_catboost(catboost_config_for_each_type[type_enc], train_X, train_Y, test_X, test_Y,
                                  train_X_df.shape[0])
        perm_results, bad_features = permutation_importance(model, test_X, test_Y, one_type_eval_metric, verbose=False)
        print([('base', one_type_eval_metric(test_Y, model.predict(test_X)))] + [(bf, perm_results[bf])
                                                                                 for bf in bad_features])
        bad_features_dict[type_enc] = bad_features

    return bad_features


def train_for_each_type_no_model(model_config_for_each_type,
                                 useless_cols_for_each_type,
                                 train_X_df,
                                 Y_df,
                                 train_fn=None,
                                 test_X_df=None):
    """
    It does not store (and return) model. It is very lean in terms of memory in that sense.
    """
    if train_fn is None:
        train_fn = train_catboost

    anal_dict = {}
    predictions = []
    for type_enc in train_X_df['type_enc'].unique():
        print(type_enc)
        anal_dict[type_enc] = {}
        X_t = train_X_df[train_X_df.type_enc == type_enc]

        if len(useless_cols_for_each_type[type_enc]) > 0:
            X_t = X_t.drop(useless_cols_for_each_type[type_enc], axis=1)

        if test_X_df is not None:
            train_X = X_t
            test_X = test_X_df[test_X_df.type_enc == type_enc][train_X.columns]
            test_Y = None
        else:
            train_X, test_X = train_test_split(X_t, test_size=0.15, random_state=0)
            test_Y = Y_df.loc[test_X.index]

        train_Y = Y_df.loc[train_X.index]
        model, feature_importance_df, best_iteration = train_fn(model_config_for_each_type[type_enc], train_X, train_Y,
                                                                test_X, test_Y, train_X_df.shape[0])
        # saving important things.
        anal_dict[type_enc]['train'] = pd.Series(model.predict(train_X), index=train_X.index)
        anal_dict[type_enc]['best_iteration'] = best_iteration
        anal_dict[type_enc]['feature_importance'] = feature_importance_df['Importances'].to_dict()

        print('Best iter', best_iteration, model_config_for_each_type[type_enc])

        anal_dict[type_enc]['test'] = pd.Series(model.predict(test_X), index=test_X.index)
        predictions.append(anal_dict[type_enc]['test'])
        if test_Y is not None:
            print('Eval', eval_metric(test_Y.values, predictions[-1].values, test_X['type_enc']))

    prediction_df = pd.concat(predictions)
    if test_X_df is None:
        actual = Y_df.loc[prediction_df.index]
        print('Final metric', eval_metric(actual.values, prediction_df.values,
                                          train_X_df.loc[actual.index, 'type_enc']))

    return anal_dict


def train_for_each_contrib(catboost_config, train_X_df, Y_df):
    train_X, test_X = train_test_split(train_X_df, test_size=0.15, random_state=0)
    test_Y = Y_df.loc[test_X.index]

    train_Y = Y_df.loc[train_X.index]
    prediction = 0
    for col in Y_df.colums:
        model = train_catboost(catboost_config, train_X, train_Y[col], test_X, test_Y[col], train_X_df.shape[0])
        prediction += model.predict(test_X)

    actual = Y_df.sum(axis=1)
    print(one_type_eval_metric(actual, prediction))


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
        models[type_enc] = train_catboost(catboost_config_for_each_type[type_enc], train_X, train_Y, test_X, test_Y,
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
