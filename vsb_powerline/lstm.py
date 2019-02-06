import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold


# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


class LSTModel:
    def __init__(self, units, dense_count):
        self._units = units
        self._dense_c = dense_count
        self._data_dir = '/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/'
        self._data_fname = 'transformed_train.csv'
        self._meta_fname = 'metadata_train.csv'

        self._n_splits = 5
        self._feature_c = None
        self._ts_c = None

    def get_model(self):
        inp = Input(shape=(
            self._ts_c,
            self._feature_c,
        ))
        x = LSTM(self._units, return_sequences=True)(inp)
        x = LSTM(self._units // 2, return_sequences=False)(x)
        x = Dense(self._dense_c, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
        return model

    def get_processed_train_df(self):
        fname = self._data_dir + self._data_fname
        processed_train_df = pd.read_csv(fname, compression='gzip', index_col=[0, 1])
        processed_train_df = processed_train_df.T
        processed_train_df = processed_train_df.swaplevel(axis=1).sort_index(axis=1)
        processed_train_df = processed_train_df.drop('Unnamed: 0', axis=0)
        processed_train_df.index = list(map(int, processed_train_df.index))
        return processed_train_df.sort_index(axis=0)

    def get_y(self):
        fname = self._data_dir + self._meta_fname
        df = pd.read_csv(fname)
        return df.set_index('signal_id')

    def get_X_y(self):
        """
        X.shape should be: (#examples,#ts,#features)
        y.shape should be: (#examples,)
        """
        processed_train_df = self.get_processed_train_df()
        # NOTE: there are 8 columns which are being zero. one needs to fix it.
        assert processed_train_df.isna().any(axis=0).sum() == 8
        assert processed_train_df.isna().all(axis=0).sum() == 8

        processed_train_df = processed_train_df.fillna(0)
        assert not processed_train_df.isna().any().any(), 'Training data has nan'

        y_df = self.get_y()
        y_df = y_df.loc[processed_train_df.index]

        examples_c = processed_train_df.shape[0]
        self._ts_c = len(processed_train_df.columns.levels[0])
        self._feature_c = len(processed_train_df.columns.levels[1])

        print('#examples', examples_c)
        print('#ts', self._ts_c)
        print('#features', self._feature_c)
        print('data shape', processed_train_df.shape)
        X = processed_train_df.values.reshape(examples_c, self._ts_c, self._feature_c)
        y = y_df.target.values

        assert X.shape == (examples_c, self._ts_c, self._feature_c)
        assert y.shape == (examples_c, )
        return X, y

    def train(self):
        X, y = self.get_X_y()
        print('X shape', X.shape)
        print('Y shape', y.shape)

        splits = list(StratifiedKFold(n_splits=self._n_splits, shuffle=True).split(X, y))

        preds_val = []
        y_val = []
        # Then, iteract with each fold
        # If you dont know, enumerate(['a', 'b', 'c']) returns [(0, 'a'), (1, 'b'), (2, 'c')]
        for idx, (train_idx, val_idx) in enumerate(splits):
            K.clear_session()  # I dont know what it do, but I imagine that it "clear session" :)
            print("Beginning fold {}".format(idx + 1))
            # use the indexes to extract the folds in the train and validation data
            train_X, train_y, val_X, val_y = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

            print('Train X shape', train_X.shape)
            print('Val X shape', val_X.shape)
            print('Train Y shape', train_y.shape)
            print('Val y shape', val_y.shape)

            # instantiate the model for this fold
            model = self.get_model()
            print(model.summary())
            # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
            # validation matthews_correlation greater than the last one.
            ckpt = ModelCheckpoint(
                'weights_{}.h5'.format(idx),
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                monitor='val_matthews_correlation',
                mode='max',
            )

            # Train, train, train
            model.fit(train_X, train_y, batch_size=20, epochs=100, validation_data=[val_X, val_y], callbacks=[ckpt])
            # loads the best weights saved by the checkpoint
            model.load_weights('weights_{}.h5'.format(idx))
            # Add the predictions of the validation to the list preds_val
            preds_val.append(model.predict(val_X, batch_size=20))
            # and the val true y
            y_val.append(val_y)


if __name__ == '__main__':
    model = LSTModel(64, 16)
    model.train()
