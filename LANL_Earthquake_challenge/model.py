from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense, SimpleRNN
from keras.callbacks import ModelCheckpoint
from data import Data


class Model:
    def __init__(
            self,
            ts_window: int,
            ts_size: int,
            train_fname: str,
    ):
        self._ts_window = ts_window
        self._ts_size = ts_size
        self._hidden_lsize = 60
        self._model = None
        self._model = None
        self._history = None

        self._data_cls = Data(self._ts_window, self._ts_size, train_fname)
        self._X_val, self._y_val = self._data_cls.get_validation_X_y()

    def get_model(self, feature_count, learning_rate: float):
        model = Sequential()
        model.add(CuDNNGRU(self._hidden_lsize, input_shape=(self._ts_window, feature_count)))
        model.add(Dense(self._hidden_lsize // 2, activation='relu'))
        model.add(Dense(1))
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

    def fit(self, epochs: int, learning_rate: float):
        feature_count = self._X_val.shape[2]
        self._model = self.get_model(feature_count, learning_rate=learning_rate)
        steps_per_epoch = int(self._data_cls.training_size() / self._data_cls.batch_size())

        ckpt = ModelCheckpoint(
            'weights.h5',
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
            monitor='val_loss',
            mode='min',
        )

        # Train
        self._history = self._model.fit_generator(
            self._data_cls.get_X_y_generator(),
            epochs=epochs,
            validation_data=[self._X_val, self._y_val],
            callbacks=[ckpt],
            steps_per_epoch=steps_per_epoch,
            # workers=2,
            # use_multiprocessing=True,
            verbose=2,
        )

        self._plot_acc_loss()
        # load the weights which were best for validation.
        self._model.load_weights('weights.h5')

    def predict(self, df):
        X = self._data_cls.get_test_X(df)
        return self._model.predict(X)


if __name__ == '__main__':
    ts_window = 50
    ts_size = 1000
    model = Model(ts_window, ts_size, 'train.csv')
    epochs = 20
    model.fit(epochs, 0.0005)
