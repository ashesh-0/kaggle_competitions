X_train = X_t.iloc[train_idx]
Y_train = Y_t.loc[X_train.index]

X_test = X_t.iloc[test_idx]
Y_test = Y_t.loc[X_test.index]

def train_model(X_train, Y_train, X_test, Y_test):
    model = CatboostModel()
    model.fit(X_train, Y_train, X_test, Y_test)
    models[molecule_type].append(model)
    predictions[molecule_type].append(pd.Series(model.predict(X_test), index=X_test.index))

    tr_metric, test_metric = get_performance(model, X_train, Y_train, X_test, Y_test, raw_train_df)