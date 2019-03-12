from typing import Tuple
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (Bidirectional, Dense, Input, CuDNNLSTM, Activation, BatchNormalization, LeakyReLU, Dropout)
from keras.models import Model
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from keras import regularizers

# from threadsafe_iterator import threadsafe


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
            units: int,
            dense_count: int,
            train_fname='/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/transformed_train.csv',
            meta_train_fname='/home/ashesh/Documents/initiatives/kaggle_competitions/vsb_powerline/data/metadata_train.csv',
            skip_fraction: float = 0,
            data_aug_num_shifts=1,
            data_aug_flip=False,
            dropout_fraction=0.3,
            add_other_phase_data=False,
            same_prediction_over_id_measurement=False,
            plot_stats=True):
        """
        Args:
            skip_fraction: initial fraction of timestamps can be ignored.
        """
        self._units = units
        self._dense_c = dense_count
        self._data_fname = train_fname
        self._meta_fname = meta_train_fname
        self._skip_fraction = skip_fraction
        self._data_aug_num_shifts = data_aug_num_shifts
        self._data_aug_flip = data_aug_flip
        self._plot_stats = plot_stats
        self._dropout_fraction = dropout_fraction
        self._add_other_phase_data = add_other_phase_data
        self._same_prediction_over_id_measurement = same_prediction_over_id_measurement
        self._skip_features = [
            'diff_smoothend_by_1 Quant-0.25', 'diff_smoothend_by_1 Quant-0.75', 'diff_smoothend_by_1 abs_mean',
            'diff_smoothend_by_1 mean', 'diff_smoothend_by_16 Quant-0.25', 'diff_smoothend_by_16 Quant-0.75',
            'diff_smoothend_by_16 abs_mean', 'diff_smoothend_by_16 mean', 'diff_smoothend_by_2 Quant-0.25',
            'diff_smoothend_by_2 Quant-0.75', 'diff_smoothend_by_4 Quant-0.25', 'diff_smoothend_by_4 Quant-0.75',
            'diff_smoothend_by_8 Quant-0.25', 'diff_smoothend_by_8 Quant-0.5', 'signal_Quant-0.25', 'signal_Quant-0.75'
        ]

        # # their distribution is significantly different in test data from train.
        # self._skip_features += [
        #     'diff_smoothend_by_2 Quant-0.0', 'signal_Quant-0.5', 'diff_smoothend_by_4 Quant-0.0', 'peak_width_1',
        #     'peak_width_0', 'diff_smoothend_by_8 Quant-0.0', 'diff_smoothend_by_16 Quant-0.0', 'peak_distances_1',
        #     'peak_distances_0.75', 'peak_distances_0.5'
        # ]

        self._n_splits = 3
        self._feature_c = None
        self._ts_c = None
        # validation score is saved here.
        self._val_score = -1
        # normalization is done using this.
        self._n_split_scales = []
        # a value between 0 and 1. a prediction greater than this value is considered as 1.
        self.threshold = None

    def get_model(self):
        inp = Input(shape=(
            self._ts_c,
            self._feature_c,
        ))
        x = Bidirectional(
            CuDNNLSTM(
                self._units,
                return_sequences=False,
                #                 kernel_regularizer=regularizers.l1(0.001),
                # activity_regularizer=regularizers.l1(0.01),
                # bias_regularizer=regularizers.l1(0.01)
            ))(inp)

        #         x = Bidirectional(CuDNNLSTM(self._units, return_sequences=False))(inp)
        #         x = Bidirectional(CuDNNLSTM(self._units // 2, return_sequences=False,
        #                                    kernel_regularizer=regularizers.l1(0.001),))(x)
        x = Dropout(self._dropout_fraction)(x)
        x = Dense(self._dense_c)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[matthews_correlation])
        return model

    @staticmethod
    def skip_quantile_features(cols, quantiles):
        filt_cols1 = LSTModel._skip_quantiles(cols, quantiles, '_')
        filt_cols2 = LSTModel._skip_quantiles(cols, quantiles, '-')
        cols3 = list(set(filt_cols1) & set(filt_cols2))
        cols3.sort()
        return cols3

    @staticmethod
    def _skip_quantiles(cols, quantiles, delimiter):
        filtered_cols = []
        for col in cols:
            try:
                val = float(col.split(delimiter)[-1])
                if val in quantiles:
                    continue
            except:
                pass
            filtered_cols.append(col)
        return filtered_cols

    def get_processed_data_df(self, fname: str):
        processed_data_df = pd.read_csv(fname, compression='gzip', index_col=[0, 1])
        processed_data_df = processed_data_df.T
        processed_data_df = processed_data_df.swaplevel(axis=1).sort_index(axis=1)
        if 'Unnamed: 0' in processed_data_df.index:
            processed_data_df = processed_data_df.drop('Unnamed: 0', axis=0)

        processed_data_df.index = list(map(int, processed_data_df.index))

        # skip unnecessary columns
        # feature_cols = LSTModel.skip_quantile_features(processed_data_df.columns.levels[1], [0.25, 0.75])
        feature_cols = list(set(processed_data_df.columns.levels[1]) - set(self._skip_features))
        processed_data_df = processed_data_df.iloc[:, processed_data_df.columns.get_level_values(1).isin(feature_cols)]
        processed_data_df.columns = processed_data_df.columns.remove_unused_levels()

        # skip first few timestamps. (from paper.)
        ts_units = len(processed_data_df.columns.levels[0])
        skip_end_ts_index = int(ts_units * self._skip_fraction) - 1
        if skip_end_ts_index > 0:
            print('Skipping first ', skip_end_ts_index + 1, 'timestamp units out of total ', ts_units, ' units')
            col_filter = processed_data_df.columns.get_level_values(0) > skip_end_ts_index
            processed_data_df = processed_data_df.iloc[:, col_filter]

        return processed_data_df.sort_index(axis=0).sort_index(axis=1)

    def get_y_df(self):
        fname = self._meta_fname
        df = pd.read_csv(fname)
        return df.set_index('signal_id')

    def add_phase_data(self, processed_data_df, meta_fname):
        """
        Args:
            processed_data_df: 2 level columns. level 0 is timestamp(int). level 1 is features.(str)
            meta_fname: meta file
        """
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

        assert 'id_measurement' not in data_1.columns.levels[0]
        assert 'id_measurement' not in data_2.columns.levels[0]
        assert 'id_measurement' not in data_3.columns.levels[0]
        assert 'id_measurement' not in data_1.columns.levels[1]
        assert 'id_measurement' not in data_2.columns.levels[1]
        assert 'id_measurement' not in data_3.columns.levels[1]

        # change indicators name to ensure uniqueness of columns
        feat_names = ['Phase1-' + e for e in data_1.columns.levels[1].tolist()]
        data_1.columns.set_levels(feat_names, level=1, inplace=True)

        feat_names = ['Phase2-' + e for e in data_2.columns.levels[1].tolist()]
        data_2.columns.set_levels(feat_names, level=1, inplace=True)

        feat_names = ['Phase3-' + e for e in data_3.columns.levels[1].tolist()]
        data_3.columns.set_levels(feat_names, level=1, inplace=True)

        processed_data_df = pd.concat([data_1, data_2, data_3], axis=1).sort_index(axis=1)
        print(processed_data_df.shape)
        print('Phase data added')
        return processed_data_df

    def get_X_df(self, fname, meta_fname):
        processed_data_df = self.get_processed_data_df(fname)

        # NOTE: there are 8 columns which are being zero. one needs to fix it.
        assert processed_data_df.isna().any(axis=0).sum() <= 8
        assert processed_data_df.isna().all(axis=0).sum() <= 8

        processed_data_df = processed_data_df.fillna(0)
        if self._add_other_phase_data:
            processed_data_df = self.add_phase_data(processed_data_df, meta_fname)

        assert not processed_data_df.isna().any().any(), 'Training data has nan'
        return processed_data_df

    def get_X_in_parts_df(self, fname, meta_fname):
        processed_data_df = self.get_processed_data_df(fname)

        # NOTE: there are 8 columns which are being zero. one needs to fix it.
        assert processed_data_df.isna().any(axis=0).sum() <= 8
        assert processed_data_df.isna().all(axis=0).sum() <= 8

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

            assert set(meta_df.iloc[s_index:e_index]['id_measurement'].value_counts().values.tolist()) == set([3])

            # making all three phases data available.
            data_df = processed_data_df.iloc[s_index:e_index]
            if self._add_other_phase_data:
                data_df = self.add_phase_data(data_df, meta_fname)

            assert not data_df.isna().any().any(), 'Training data has nan'
            s_index = e_index
            e_index = s_index + chunksize
            print('Completed Test data preprocessing', round(e_index / sz * 100), '%')
            yield data_df

        data_df = processed_data_df.iloc[s_index:]
        if self._add_other_phase_data:
            data_df = self.add_phase_data(data_df, meta_fname)

        assert not data_df.isna().any().any(), 'Training data has nan'
        yield data_df

    def get_X_y(self):
        """
        Returns:
            Tuple(X,y):
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

    def predict(self, fname: str, meta_fname: str):
        ser = self._predict(fname, meta_fname)
        ser.index.name = 'signal_id'

        if self._same_prediction_over_id_measurement:
            ser = ser.to_frame('prediction')
            meta_df = pd.read_csv(meta_fname).set_index('signal_id')
            df = ser.join(meta_df[['id_measurement']], how='left')
            ser = df.groupby('id_measurement').transform(np.mean)['prediction']
            ser[ser >= 0.5] = 1
            ser[ser < 0.5] = 0
            ser = ser.astype(int)

        return ser

    def _predict(self, fname: str, meta_fname):
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

                scale = self._n_split_scales[split_index]
                pred_array.append(model.predict(X / scale, batch_size=128))

            # Take average value over different models.
            pred_array = np.array(pred_array).reshape(len(pred_array), -1)
            pred_array = (pred_array > self.threshold).astype(int)
            pred = np.mean(np.array(pred_array), axis=0)
            # majority prediction
            pred = (pred > 0.5).astype(int)
            assert pred.shape[0] == X.shape[0]

            output.append(pred)
            output_index.append(df.index.tolist())
        return pd.Series(np.squeeze(np.concatenate(output)), index=np.concatenate(output_index))

    def fit_threshold(self, prediction, actual, start=0.08, end=0.98, n_count=20, center_alignment_offset=0.01):
        best_score = -1
        self.threshold = 0
        scores = []
        thresholds = np.linspace(start, end, n_count)
        for threshold in thresholds:
            score = matthews_corrcoef(actual, (prediction > threshold).astype(np.float64))
            scores.append(score)
            center_alignment = 1 if threshold > (1 - self.threshold) else -1
            if score > best_score + center_alignment * center_alignment_offset:
                best_score = score
                self.threshold = threshold

        if self._plot_stats:
            import matplotlib.pyplot as plt

            plt.plot(thresholds, scores)
            plt.title('Fitting threshold')
            plt.ylabel('mathews correlation coef')
            plt.xlabel('threshold')
            plt.show()

        print('Matthews correlation on train set is ', best_score, ' with threshold:', self.threshold)

    @staticmethod
    def get_generator(train_X: np.array, train_y: np.array, batch_size: int, flip: bool, num_shifts: int = 2):

        shifts = list(map(int, np.linspace(0, train_X.shape[1] * 0.1, num_shifts + 1)[1:-1]))
        shifts = [0] + shifts
        flip_ts = [1, -1] if flip else [1]

        @threadsafe
        def augument_by_timestamp_shifts() -> Tuple[np.array, np.array]:
            """
            num_shifts: factor by which the training data is to be increased.
            We shift the timestamps to get more data to train. It assumes timestamp is in 2nd dimension of
            train_X
            """
            num_times = len(flip_ts) * num_shifts
            print('After data augumentation, training data has become ', num_times, ' times its original size.')
            # generator = DataGenerator('training_data_augumented.csv', batch_size, train_X.shape[1], train_X.shape[2],)
            # generator.add(train_X, train_y)
            # 1 time is the original data itself.
            while True:
                for shift_amount in shifts:
                    for flip_direction in flip_ts:
                        flipped_X = train_X[:, ::flip_direction, :]
                        train_X_shifted = np.roll(flipped_X, shift_amount, axis=1)
                        for index in range(0, train_X_shifted.shape[0], batch_size):
                            X = train_X_shifted[index:(index + batch_size), :, :]
                            y = train_y[index:(index + batch_size)]
                            yield (X, y)

        steps_per_epoch = len(flip_ts) * len(shifts) * train_X.shape[0] // batch_size
        return augument_by_timestamp_shifts, steps_per_epoch

    def _plot_acc_loss(self, history):
        import matplotlib.pyplot as plt

        plt.plot(history.history['matthews_correlation'])
        plt.plot(history.history['val_matthews_correlation'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

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

            # We should get scale using normalization on train data only.
            # axis 0 is #examples, 1 is #timestamps, 2 is features.
            scale = np.abs(np.max(train_X, axis=(0, 1)))
            scale[scale == 0] = 1

            train_X = train_X / scale
            val_X = val_X / scale
            self._n_split_scales.append(scale)

            # data augumentation
            generator, steps_per_epoch = LSTModel.get_generator(
                train_X,
                train_y,
                batch_size,
                self._data_aug_flip,
                num_shifts=self._data_aug_num_shifts,
            )
            # print('Train X shape', train_X.shape)
            # print('Val X shape', val_X.shape)
            # print('Train Y shape', train_y.shape)
            # print('Val y shape', val_y.shape)

            model = self.get_model()
            print(model.summary())
            # This checkpoint helps to avoid overfitting. It just save the weights of the model if it delivered an
            # validation matthews_correlation greater than the last one.
            ckpt = ModelCheckpoint(
                'weights_{}.h5'.format(idx),
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
                monitor='val_matthews_correlation',
                mode='max',
            )

            # Train
            history = model.fit_generator(
                generator(),
                epochs=epoch,
                validation_data=[val_X, val_y],
                callbacks=[ckpt],
                steps_per_epoch=steps_per_epoch,
                # workers=2,
                # use_multiprocessing=True,
                verbose=0,
            )

            if self._plot_stats:
                self._plot_acc_loss(history)

            # loads the best weights saved by the checkpoint
            model.load_weights('weights_{}.h5'.format(idx))
            # Add the predictions of the validation to the list preds_val
            preds_array.append(model.predict(val_X, batch_size=20))
            y_array.append(val_y)

        prediction = np.concatenate(preds_array)
        actual = np.concatenate(y_array)

        self.fit_threshold(model.predict(train_X), train_y)
        self._val_score = matthews_corrcoef(actual, (prediction > self.threshold).astype(np.float64))
        print('On validation data, score is:', self._val_score)
