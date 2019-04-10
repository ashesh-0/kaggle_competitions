from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense, SimpleRNN
from data import Data


class Model:
    def __init__(self, ts_window: int, ts_size: int, learning_rate: float = 0.0005):
        self._ts_window = ts_window
        self._ts_size = ts_size
        self._hidden_lsize = 60
        self._learning_rate = learning_rate
        self._model = None
        self._data_cls = None
        self._model = None
        self._history = None

    def get_model(self, feature_count):
        model = Sequential()
        model.add(SimpleRNN(self._hidden_lsize, input_shape=(self._ts_window, feature_count)))
        model.add(Dense(self._hidden_lsize // 2, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=adam(lr=self._learning_rate), loss="mae")
        print(model.summary())
        return model

    def fit(self, fname, epochs):
        self._data_cls = Data(self._ts_window, self._ts_size, fname)
        X_val, y_val = self._data_cls.get_validation_X_y()
        feature_count = X_val.shape[2]
        self._model = self.get_model(feature_count)
        steps_per_epoch = int(self._data_cls.training_size() / self._data_cls.batch_size())
        # Train
        self._history = self._model.fit_generator(
            self._data_cls.get_X_y_generator(),
            epochs=epochs,
            validation_data=[X_val, y_val],
            # callbacks=[ckpt],
            steps_per_epoch=steps_per_epoch,
            # workers=2,
            # use_multiprocessing=True,
            verbose=2,
        )

    def predict(self, df):
        X = self._data_cls.get_test_X(df)
        return self._model.predict(X)


# if __name__ == '__main__':
#     ts_window = 50
#     ts_size = 1000
#     model = Model(ts_window, ts_size)
#     epochs = 20
#     model.fit('train.csv', epochs)
