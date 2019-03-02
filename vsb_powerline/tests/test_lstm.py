import numpy as np
from lstm import LSTModel


def _check_new_X(X, new_X, n_times):
    assert new_X.shape == (n_times * X.shape[0], X.shape[1], X.shape[2])
    assert (new_X[:X.shape[0], :, :] == X).all()
    shifts = []
    for i in range(X.shape[0], X.shape[0] * n_times, X.shape[0]):
        found = False
        for shift in range(1, X.shape[2]):
            guess = np.roll(X, shift, axis=2)
            if (guess == new_X[i:i + X.shape[0], :, :]).all():
                shifts.append(shift)
                found = True
                break
        assert found, 'No shift was found which matches the synthesized data.'

    assert len(set(shifts)) == n_times - 1, 'Duplicate shifts exist'


def test_augument_by_timestamp_shifts_2Dy():
    X = np.arange(100).reshape(4, 1, 25)
    y = (np.random.rand(4, 1) > 1).astype(int)
    n_times = 5
    generator, steps = LSTModel.get_generator(X, y, 2, n_times=n_times)
    new_Xs = []
    new_ys = []
    i = 0
    for new_X_chunk, new_y_chunk in generator():
        new_Xs.append(new_X_chunk)
        new_ys.append(new_y_chunk)
        i += 1
        if i == steps:
            break

    new_X = np.concatenate(new_Xs)
    new_y = np.concatenate(new_ys)

    assert (np.tile(y, (n_times, 1)) == new_y).all()
    _check_new_X(X, new_X, n_times)


def test_augument_by_timestamp_shifts_1Dy():
    X = np.arange(100).reshape(4, 1, 25)
    y = (np.random.rand(4, 1) > 1).astype(int).reshape(X.shape[0], )
    n_times = 5
    generator, steps = LSTModel.get_generator(X, y, 2, n_times=n_times)
    new_Xs = []
    new_ys = []
    i = 0
    for new_X_chunk, new_y_chunk in generator():
        new_Xs.append(new_X_chunk)
        new_ys.append(new_y_chunk)
        i += 1
        if i == steps:
            break

    new_X = np.concatenate(new_Xs)
    new_y = np.concatenate(new_ys)

    assert (np.tile(y, (n_times)) == new_y).all()
    _check_new_X(X, new_X, n_times)
