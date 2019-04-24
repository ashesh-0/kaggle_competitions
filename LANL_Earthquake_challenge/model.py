import pandas as pd
import pickle
from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense, SimpleRNN, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from data import Data
from keras import regularizers
from typing import List


class Model:
    def __init__(
            self,
            ts_window: int,
            ts_size: int,
            train_fname: str,
            data_pickle_file: str = None,
    ):
        self._ts_window = ts_window
        self._ts_size = ts_size
        self._model = None
        self._model = None
        self._history = None

        if data_pickle_file is not None:
            print('[Model] loading from pickle file', data_pickle_file)
            with open(data_pickle_file, 'rb') as f:
                self._data_cls = pickle.load(f)
        else:
            self._data_cls = Data(self._ts_window, self._ts_size, train_fname)

        print('[Model] Validation has shape', self._data_cls.val_X.shape)

    def get_model(
            self,
            hidden_lsizes: List[int],
            feature_count: int,
            learning_rate: float,
            l1_regularizer_wt: float,
            dropout_fraction: float,
            batch_normalization: bool,
    ):
        model = Sequential()
        model.add(
            CuDNNGRU(
                hidden_lsizes[0],
                input_shape=(self._ts_window, feature_count),
                kernel_regularizer=regularizers.l1(l1_regularizer_wt),
                activation='relu'
            ))
        if batch_normalization:
            model.add(BatchNormalization())

        for hidden_lsize in hidden_lsizes[1:]:
            model.add(Dense(hidden_lsize, activation='relu'))

            if batch_normalization:
                model.add(BatchNormalization())

            model.add(Dropout(dropout_fraction))

        model.add(Dense(1, activation=None))
        model.compile(optimizer=adam(lr=learning_rate), loss="mae")
        print(model.summary())
        return model

    def _plot_acc_loss(self):
        import matplotlib.pyplot as plt
        # summarize history for loss
        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def fit(self, hidden_lsizes: List[int], epochs: int, learning_rate: float, l1_regularizer_wt: float,
            dropout_fraction: float, batch_normalization: bool, tensorboard_log_dir: str=None,):
        feature_count = self._data_cls.val_X.shape[2]
        self._model = self.get_model(hidden_lsizes, feature_count, learning_rate, l1_regularizer_wt, dropout_fraction,
                                     batch_normalization)
        steps_per_epoch = int(self._data_cls.training_size() / self._data_cls.batch_size())

        ckpt = ModelCheckpoint(
            'weights.h5',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
            monitor='val_loss',
            mode='min',
        )


        callbacks=[ckpt]
        if tensorboard_log_dir is not None:
            tboard = TensorBoard(tensorboard_log_dir, histogram_freq=1, write_grads=True)
            callbacks.append(tboard)

        # Train
        self._history = self._model.fit_generator(
            self._data_cls.get_X_y_generator(),
            epochs=epochs,
            validation_data=[self._data_cls.val_X, self._data_cls.val_y],
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch,
            # workers=2,
            # use_multiprocessing=True,
            verbose=2,
        )

        # load the weights which were best for validation.
        self._model.load_weights('weights.h5')
        self._plot_acc_loss()

    def plot_prediction(self):
        prediction = self._model.predict(self._data_cls.val_X).reshape(-1, )
        actual = self._data_cls.val_y
        title = 'Plot of prediction on validation set'
        pd.DataFrame(list(zip(actual, prediction)), columns=['actual', 'prediction']).plot(title=title)

    def predict(self, df):
        X = self._data_cls.get_test_X(df)
        return self._model.predict(X)


if __name__ == '__main__':
    ts_window = 50
    ts_size = 1000
    model = Model(ts_window, ts_size, 'train.csv')
    epochs = 10
    hidden_lsizes = [64, 32, 32]
    log_dir = '/home/ashesh/Documents/initiatives/kaggle_competitions/LANL_Earthquake_challenge/log'
    model.fit(hidden_lsizes, epochs, 0.00001, 0.001, 0.2, True, log_dir)
