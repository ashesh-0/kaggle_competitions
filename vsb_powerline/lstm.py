import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef


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
    def __init__(
            self,
            units,
            dense_count,
            train_fname='/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/transformed_train.csv',
            meta_train_fname='/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/metadata_train.csv',
    ):
        self._units = units
        self._dense_c = dense_count
        self._data_fname = train_fname
        self._meta_fname = meta_train_fname

        self._n_splits = 5
        self._feature_c = None
        self._ts_c = None
        # a value between 0 and 1. a prediction greater than this value is considered as 1.
        self.threshold = None

    def get_model(self):
        inp = Input(shape=(
            self._ts_c,
            self._feature_c,
        ))
        x = LSTM(self._units, return_sequences=False)(inp)
        # x = LSTM(self._units // 2, return_sequences=False)(x)
        x = Dense(self._dense_c, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
        return model

    def get_processed_data_df(self, fname: str):
        processed_data_df = pd.read_csv(fname, compression='gzip', index_col=[0, 1])
        processed_data_df = processed_data_df.T
        processed_data_df = processed_data_df.swaplevel(axis=1).sort_index(axis=1)
        if 'Unnamed: 0' in processed_data_df.index:
            processed_data_df = processed_data_df.drop('Unnamed: 0', axis=0)

        processed_data_df.index = list(map(int, processed_data_df.index))
        return processed_data_df.sort_index(axis=0)

    def get_y_df(self):
        fname = self._meta_fname
        df = pd.read_csv(fname)
        return df.set_index('signal_id')

    def add_phase_data(self, processed_data_df, meta_fname):
        print('Phase data is about to be added')
        metadata_df = pd.read_csv(meta_fname).set_index('signal_id')
        processed_data_df = processed_data_df.join(metadata_df[['id_measurement']], how='left')
        assert not processed_data_df.isna().any().any()

        # pandas does not have a cyclic shift facility. therefore copying it.
        temp_df = pd.concat([processed_data_df, processed_data_df])
        grp = temp_df.groupby('id_measurement')

        data_1 = grp.shift(0)
        data_1 = data_1[~data_1.index.duplicated(keep='first')]

        data_2 = grp.shift(-1)
        data_2 = data_2[~data_2.index.duplicated(keep='first')]

        data_3 = grp.shift(-2)
        data_3 = data_3[~data_3.index.duplicated(keep='first')]
        del grp
        del temp_df

        assert set(data_1.index.tolist()) == (set(data_2.index.tolist()))
        assert set(data_1.index.tolist()) == (set(data_3.index.tolist()))

        # change indicators name to ensure uniqueness of columns
        feat_names = ['Phase1-' + e for e in data_1.columns.levels[1].tolist()]
        data_1.columns.set_levels(feat_names, level=1, inplace=True)

        feat_names = ['Phase2-' + e for e in data_2.columns.levels[1].tolist()]
        data_2.columns.set_levels(feat_names, level=1, inplace=True)

        feat_names = ['Phase3-' + e for e in data_3.columns.levels[1].tolist()]
        data_3.columns.set_levels(feat_names, level=1, inplace=True)

        processed_data_df = pd.concat([data_1, data_2, data_3], axis=1)
        print(processed_data_df.shape)
        print('Phase data added')
        return processed_data_df

    def get_X_df(self, fname, meta_fname):
        processed_data_df = self.get_processed_data_df(fname)

        # NOTE: there are 8 columns which are being zero. one needs to fix it.
        assert processed_data_df.isna().any(axis=0).sum() == 8
        assert processed_data_df.isna().all(axis=0).sum() == 8

        processed_data_df = processed_data_df.fillna(0)
        processed_data_df = self.add_phase_data(processed_data_df, meta_fname)
        assert not processed_data_df.isna().any().any(), 'Training data has nan'
        return processed_data_df

    def get_X_in_parts_df(self, fname, meta_fname):
        processed_data_df = self.get_processed_data_df(fname)

        # NOTE: there are 8 columns which are being zero. one needs to fix it.
        assert processed_data_df.isna().any(axis=0).sum() == 8
        assert processed_data_df.isna().all(axis=0).sum() == 8

        processed_data_df = processed_data_df.fillna(0)
        meta_df = pd.read_csv(meta_fname)
        chunksize = 2 * 999
        s_index = 0
        e_index = chunksize
        sz = processed_data_df.shape[0]
        while e_index < sz:
            last_accesible_id = meta_df.iloc[e_index - 1]['id_measurement']
            first_inaccesible_id = meta_df.iloc[e_index]['id_measurement']
            while e_index < sz and last_accesible_id == first_inaccesible_id:
                e_index += 1
                last_accesible_id = meta_df.iloc[e_index - 1]['id_measurement']
                first_inaccesible_id = meta_df.iloc[e_index]['id_measurement']

            # making all three phases data available.
            data_df = self.add_phase_data(processed_data_df.iloc[s_index:e_index], meta_fname)
            assert not data_df.isna().any().any(), 'Training data has nan'
            s_index = e_index
            e_index = s_index + chunksize
            print('Completed Test data preprocessing', round(e_index / sz * 100), '%')
            yield data_df

        data_df = self.add_phase_data(processed_data_df.iloc[s_index:], meta_fname)
        assert not data_df.isna().any().any(), 'Training data has nan'
        yield data_df

    def get_X_y(self):
        """
        X.shape should be: (#examples,#ts,#features)
        y.shape should be: (#examples,)
        """
        processed_train_df = self.get_X_df(self._data_fname, self._meta_fname)
        y_df = self.get_y_df()
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

    def predict(self, fname: str, meta_fname):
        """
        Using the self._n_splits(5) models, it returns a pandas.Series with values belonging to {0,1}
        """
        output = []
        output_index = []
        for df in self.get_X_in_parts_df(fname, meta_fname):
            examples_c = df.shape[0]
            X = df.values.reshape(examples_c, self._ts_c, self._feature_c)

            pred_array = []
            for split_index in range(self._n_splits):
                weight_fname = 'weights_{}.h5'.format(split_index)
                model = self.get_model()
                model.load_weights(weight_fname)
                pred_array.append(model.predict(X, batch_size=128))

            # Take average value over different models.
            pred = np.mean(np.array(pred_array), axis=0)
            assert pred.shape[0] == X.shape[0]

            pred = (pred > self.threshold).astype(int)
            output.append(pred)
            output_index.append(df.index.tolist())
        return pd.Series(np.squeeze(np.concatenate(output)), index=np.concatenate(output_index))

    def fit_threshold(self, prediction, actual):
        best_score = -1
        self.threshold = None
        for threshold in np.linspace(0.01, 0.9, 1000):
            score = matthews_corrcoef(actual, (prediction > threshold).astype(np.float64))
            if score > best_score:
                best_score = score
                self.threshold = threshold
        print('Matthews correlation on dev set is ', best_score, ' with threshold:', self.threshold)

    def train(self, batch_size=128, epoch=50):
        X, y = self.get_X_y()
        print('X shape', X.shape)
        print('Y shape', y.shape)

        splits = list(StratifiedKFold(n_splits=self._n_splits, shuffle=True).split(X, y))

        preds_array = []
        y_array = []
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
            model.fit(
                train_X,
                train_y,
                batch_size=batch_size,
                epochs=epoch,
                validation_data=[val_X, val_y],
                callbacks=[ckpt],
            )

            # loads the best weights saved by the checkpoint
            model.load_weights('weights_{}.h5'.format(idx))
            # Add the predictions of the validation to the list preds_val
            preds_array.append(model.predict(val_X, batch_size=20))
            y_array.append(val_y)

        prediction = np.concatenate(preds_array)
        actual = np.concatenate(y_array)
        self.fit_threshold(prediction, actual)


if __name__ == '__main__':
    model = LSTModel(64, 16)
    model.train(epoch=5)
