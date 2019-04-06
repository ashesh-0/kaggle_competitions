from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense


class Model:
    def __init__(self, ts_count, feature_count):
        self._ts_count = ts_count
        self._feature_count = feature_count
        self._hidden_lsize = 60

    def get_model(self):
        model = Sequential()
        model.add(CuDNNGRU(self._hidden_lsize, input_shape=(None, self._feature_count)))
        model.add(Dense(self._hidden_lsize // 2, activation='relu'))
        model.add(Dense(1))
        return model

    def fit(self):
        model.compile(optimizer=adam(lr=0.0005), loss="mae")
