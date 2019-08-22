"""
Couple of functions needed for training the model.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
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
    output_dict = {
        'model': model,
        'feature_importance': feature_importance_df,
        'best_iteration': model.best_iteration,
        'train_prediction': model.predict(train_X),
        'test_prediction': model.predict(test_X),
    }
    return output_dict


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
    feature_importance_df = model.get_feature_importance(prettified=True)
    feature_importance_df.rename({'Feature Id': 'Feature Index'}, inplace=True, axis=1)

    feature_importance_df.set_index('Feature Index', inplace=True)

    output_dict = {
        'model': model,
        'feature_importance': feature_importance_df,
        'best_iteration': model.best_iteration_,
        'train_prediction': model.predict(train_X),
        'test_prediction': model.predict(test_X),
    }
    return output_dict


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

        X_val.loc[:,col] = np.random.permutation(X_val[col])
        preds = model.predict(X_val)
        results[col] = metric(y_val, preds)

        X_val.loc[:,col] = freezed_col

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


def add_one_hot_encoding(X_df, test_X_df, skip_one_hot_columns=None):
    """
    integer columns are converted to onehot columns.  This is helpful in gradient based models LR,NN.
    Returns a dict mapping old column to new column names.
    Ensure that df does not get any more columns than test_X_df.
    """
    # we might have removed some columns from training data as part of preprocessing.
    assert set(X_df.columns).issubset(set(test_X_df.columns))

    _ = _add_one_hot_encoding(X_df, skip_one_hot_columns=skip_one_hot_columns)
    _ = _add_one_hot_encoding(test_X_df, skip_one_hot_columns=skip_one_hot_columns)
    extra_cols = list(set(X_df.columns) - set(test_X_df.columns))
    if extra_cols:
        print('ONEHOT encoding extra columns getting removed:', extra_cols)
        X_df.drop(extra_cols, axis=1, inplace=True)


def _add_one_hot_encoding(df, skip_one_hot_columns=None, one_hot_columns=None):
    """
    integer columns are converted to onehot columns.  This is helpful in gradient based models LR,NN.
    Returns a dict mapping old column to new column names.
    """
    new_columns_dict = {}

    dtypes_df = df.dtypes
    if one_hot_columns is None:
        # are float but have int values
        one_hot_columns = [
            'SpinMultiplicity', 'nbr_0_SpinMultiplicity', 'nbr_1_SpinMultiplicity', 'CC_hybridization', 'nbr_0_Type',
            'nbr_1_Type'
        ]
        one_hot_columns = list(set(dtypes_df.index.tolist()).intersection(set(one_hot_columns)))
        one_hot_columns += dtypes_df[(dtypes_df == np.uint8) | (dtypes_df == np.uint16)
                                     | (dtypes_df == np.int16)].index.tolist()
    else:
        one_hot_columns = one_hot_columns.copy()

    if skip_one_hot_columns is not None:
        for skip_col in skip_one_hot_columns:
            if skip_col in one_hot_columns:
                one_hot_columns.remove(skip_col)

    columns_added_count = 0
    columns_converted = []
    for col in tqdm_notebook(one_hot_columns):
        one_hot_df = pd.get_dummies(df[col], dtype=bool)
        one_hot_df.columns = [f'ONEHOT_{col}_{one_hot_col}' for one_hot_col in one_hot_df.columns]
        new_columns_dict[col] = one_hot_df.columns.tolist()

        # We ensure that only certain number of columns gets added. Otherwise, it results in issues.
        if columns_added_count + one_hot_df.shape[1] > 200:
            break

        columns_added_count += one_hot_df.shape[1]

        for one_hot_col in one_hot_df.columns:
            df[one_hot_col] = one_hot_df[one_hot_col]

        columns_converted.append(col)
        df.drop([col], axis=1, inplace=True)

    print(f'{columns_added_count} many columns added from {len(columns_converted)} columns')
    # df.drop(columns_converted, axis=1, inplace=True)
    return new_columns_dict


def train_for_one_type_no_model_normalized_onehot(model_config, X_df, Y_df, train_fn, test_X_df):
    data = {}
    # one hot encoding
    test_X = test_X_df[X_df.columns]
    add_one_hot_encoding(X_df, test_X)
    test_X = test_X[X_df.columns]

    train_X, val_X = train_test_split(X_df, test_size=0.15, random_state=0)

    val_Y = Y_df.loc[val_X.index]
    train_Y = Y_df.loc[train_X.index]

    train_idx = train_X.index
    val_idx = val_X.index
    test_idx = test_X.index

    scalar_X = StandardScaler()
    train_X = scalar_X.fit_transform(train_X)
    val_X = scalar_X.transform(val_X)
    test_X = scalar_X.transform(test_X)

    train_dict = train_fn(model_config, train_X, train_Y, val_X, val_Y, test_X)
    # saving important things.
    data['train'] = pd.Series(train_dict['train_prediction'], index=train_idx)
    data['val'] = pd.Series(train_dict['val_prediction'], index=val_idx)
    data['test'] = pd.Series(train_dict['test_prediction'], index=test_idx)

    print('Eval', one_type_eval_metric(val_Y[val_Y.columns[0]].values, data['val'].values))
    return data


def train_for_each_type_no_model_normalized_onehot(
        model_config_for_each_type,
        useless_cols_for_each_type,
        Y_df: pd.DataFrame,
        train_fn,
):
    """
    3 Data sets: Train-Test-Validation
    It does not store (and return) model. It is very lean in terms of memory in that sense.
    Y_df has to be a dataframe. this handles the case for multiple outputs. primary target has to be the first column.
    """
    assert isinstance(Y_df, pd.DataFrame)

    anal_dict = {}
    val_predictions = []
    for type_enc in reversed(range(8)):
        print(type_enc)
        anal_dict[type_enc] = {}
        X_t = pd.read_hdf('train.hdf', f'type_enc{type_enc}')
        test_X = pd.read_hdf('test.hdf', f'type_enc{type_enc}')

        cols_to_remove = useless_cols_for_each_type[type_enc].copy()
        cols_to_remove += get_constant_features(X_t)
        valid_cols = [c for c in X_t.columns.tolist() if c not in cols_to_remove]
        print(X_t.shape[1] - len(valid_cols), 'total columns removed')
        X_t = X_t[valid_cols]
        anal_dict[type_enc] = train_for_one_type_no_model_normalized_onehot(model_config_for_each_type[type_enc], X_t,
                                                                            Y_df.loc[X_t.index], train_fn, test_X)
        pred_t_df = anal_dict[type_enc]['val'].to_frame('prediction')
        pred_t_df['type_enc'] = type_enc
        val_predictions.append(pred_t_df)

    prediction_df = pd.concat(val_predictions)
    actual = Y_df.loc[prediction_df.index, Y_df.columns[0]]
    print('Final metric', eval_metric(actual.values, prediction_df['prediction'].values, prediction_df['type_enc']))

    return anal_dict


def train_for_one_type_Kfold(
        model_config,
        train_X_df,
        Y_df: pd.DataFrame,
        train_fn,
        test_X_df=None,
        n_splits=5,
):
    """
    It does not store (and return) model. It is very lean in terms of memory in that sense.
    Y_df has to be a dataframe. this handles the case for multiple outputs. primary target has to be the first column.
    """
    assert isinstance(Y_df, pd.DataFrame)

    analysis_data = []
    type_enc = train_X_df['type_enc'].unique()
    assert len(type_enc) == 1
    type_enc = type_enc[0]
    print(type_enc)

    if test_X_df is not None:
        assert set(test_X_df['type_enc'].unique()) == set([type_enc])
        test_X_df = test_X_df[train_X_df.columns]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=955)

    train_index = train_X_df.index
    for tr_idx, val_idx in kf.split(train_X_df[[]]):
        print('')
        print(f'{len(analysis_data)}th iteration')
        train_X = train_X_df.iloc[tr_idx]
        val_X = train_X_df.iloc[val_idx]
        val_Y = Y_df.loc[val_X.index]
        train_Y = Y_df.loc[train_X.index]
        train_dict = train_fn(model_config, train_X, train_Y, val_X, val_Y, train_X_df.shape[0])
        # saving important things.
        data = {}
        data['train'] = pd.Series(train_dict['train_prediction'], index=train_index[tr_idx])
        data['val'] = pd.Series(train_dict['test_prediction'], index=train_index[val_idx])
        data['best_iteration'] = train_dict['best_iteration']
        data['feature_importance'] = train_dict['feature_importance']['Importances'].to_dict()

        print('Best iter', train_dict['best_iteration'], model_config)
        print('Eval', one_type_eval_metric(val_Y[val_Y.columns[0]].values, data['val'].values))

        if test_X_df is not None:
            data['test'] = pd.Series(train_dict['model'].predict(test_X_df), index=test_X_df.index)

        analysis_data.append(data)

    return analysis_data


def train_for_one_type_no_model(
        model_config,
        train_X_df,
        Y_df: pd.DataFrame,
        train_fn,
        test_X_df=None,
):
    """
    It does not store (and return) model. It is very lean in terms of memory in that sense.
    Y_df has to be a dataframe. this handles the case for multiple outputs. primary target has to be the first column.
    """
    assert isinstance(Y_df, pd.DataFrame)
    assert len(train_X_df['type_enc'].unique()) == 1

    data = {}

    if test_X_df is not None:
        assert set(test_X_df['type_enc'].unique()) == set(train_X_df['type_enc'].unique())

        train_X = train_X_df
        test_X = test_X_df[train_X.columns]
        test_Y = None
    else:
        train_X, test_X = train_test_split(train_X_df, test_size=0.15, random_state=0)
        test_Y = Y_df.loc[test_X.index]

    train_Y = Y_df.loc[train_X.index]
    train_idx = train_X.index
    test_idx = test_X.index

    train_dict = train_fn(model_config, train_X, train_Y, test_X, test_Y, train_X_df.shape[0])
    # saving important things.
    data['train'] = pd.Series(train_dict['train_prediction'], index=train_idx)
    data['best_iteration'] = train_dict['best_iteration']
    data['feature_importance'] = train_dict['feature_importance']['Importances'].to_dict()

    print('Best iter', train_dict['best_iteration'], model_config)

    data['test'] = pd.Series(train_dict['test_prediction'], index=test_idx)

    if test_Y is not None:
        print('Eval', one_type_eval_metric(test_Y[test_Y.columns[0]].values, data['test'].values))

    return data


def train_for_each_type_no_model(
        model_config_for_each_type,
        useless_cols_for_each_type,
        train_X_df,
        Y_df: pd.DataFrame,
        train_fn=None,
        test_X_df=None,
):
    """
    It does not store (and return) model. It is very lean in terms of memory in that sense.
    Y_df has to be a dataframe. this handles the case for multiple outputs. primary target has to be the first column.
    """
    assert isinstance(Y_df, pd.DataFrame)

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

        # TODO: move it outside.
        X_t.drop(get_constant_features(X_t), axis=1, inplace=True)
        data = train_for_one_type_no_model(
            model_config_for_each_type[type_enc],
            X_t,
            Y_df.loc[X_t.index],
            train_fn,
            test_X_df=test_X_df[test_X_df.type_enc == type_enc] if test_X_df is not None else None,
        )
        anal_dict[type_enc] = data
        predictions.append(anal_dict[type_enc]['test'])

    prediction_df = pd.concat(predictions)
    if test_X_df is None:
        actual = Y_df.loc[prediction_df.index, Y_df.columns[0]]
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


def get_constant_features(df):
    cf = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            cf.append(col)

    return cf
