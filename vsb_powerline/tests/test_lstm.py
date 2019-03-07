from unittest.mock import patch
import numpy as np
from lstm import LSTModel


def _check_new_X(X, new_X, num_times, flip: bool):
    assert new_X.shape == (num_times * X.shape[0], X.shape[1], X.shape[2])
    shifts = []
    flip_opr = [1, -1] if flip else [1]
    # for flip, we are multiplying by -1. However, this will fail for no shift case.
    epsilon = 0.001
    for i in range(0, X.shape[0] * num_times, X.shape[0]):
        found = False
        for shift in range(0, X.shape[1]):
            for flip_side in flip_opr:
                guess = np.roll(X[:, ::flip_side, :], shift, axis=1)
                if (guess == new_X[i:i + X.shape[0], :, :]).all():
                    shifts.append((shift + epsilon) * flip_side)
                    found = True
                    break
        assert found, 'No shift was found which matches the synthesized data.'

    assert len(set(shifts)) == num_times, 'Duplicate shifts exist'
    assert epsilon in shifts, 'Original data was not found'


def test_augument_by_timestamp_shifts_2Dy():
    X = np.arange(100).reshape(4, 25, 1)
    y = (np.random.rand(4, 1) > 1).astype(int)
    num_shifts = 5
    flip = False
    # number of times train data has become in size after augumentation.
    num_times = num_shifts * (1 + int(flip))
    generator, steps = LSTModel.get_generator(X, y, 2, flip, num_shifts=num_shifts)
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

    assert (np.tile(y, (num_times, 1)) == new_y).all()
    _check_new_X(X, new_X, num_times, flip)


def test_augument_by_timestamp_shifts_1Dy():
    X = np.arange(100).reshape(4, 25, 1)
    y = (np.random.rand(4, 1) > 1).astype(int).reshape(X.shape[0], )
    num_shifts = 5
    flip = False
    # number of times train data has become in size after augumentation.
    num_times = num_shifts * (1 + int(flip))

    generator, steps = LSTModel.get_generator(X, y, 2, flip, num_shifts=num_shifts)
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

    assert (np.tile(y, (num_times)) == new_y).all()
    _check_new_X(X, new_X, num_times, flip)


def test_fit_threshold():
    prediction = np.array([1, 2, 3])
    actual = prediction.copy()
    start = 0
    end = 1
    counts = 100
    best_threshold_index = 55
    early_best_threshold_index = 10
    thresholds = np.linspace(start, end, counts)

    def iter():
        while True:
            for index, _ in enumerate(thresholds):
                if index == early_best_threshold_index:
                    yield 0.8
                    continue
                if index == best_threshold_index:
                    yield 0.75
                    continue
                yield 0.6

    iterator = iter()

    def custom_score(*args, **kwargs):
        for value in iterator:
            return value

    with patch('lstm.matthews_corrcoef', side_effect=custom_score) as p:
        model = LSTModel(
            10,
            15,
            train_fname='',
            meta_train_fname='',
            skip_fraction=0,
            data_aug_num_shifts=1,
            data_aug_flip=False,
            plot_stats=False)

        model.fit_threshold(prediction, actual, start=start, end=end, n_count=counts, center_alignment_offset=0)
        assert model.threshold == thresholds[early_best_threshold_index]

        model.fit_threshold(prediction, actual, start=start, end=end, n_count=counts, center_alignment_offset=0.04)
        assert model.threshold == thresholds[early_best_threshold_index]

        model.fit_threshold(prediction, actual, start=start, end=end, n_count=counts, center_alignment_offset=0.11)
        assert model.threshold == thresholds[best_threshold_index]

        model.fit_threshold(prediction, actual, start=start, end=end, n_count=counts, center_alignment_offset=0.3)
        assert round(model.threshold, 2) == 0.49


def test_augument_by_timestamp_shifts_1Dy_with_flip():
    X = np.arange(100).reshape(4, 25, 1)
    y = (np.random.rand(4, 1) > 1).astype(int).reshape(X.shape[0], )
    num_shifts = 5
    flip = True
    # number of times train data has become in size after augumentation.
    num_times = num_shifts * (1 + int(flip))

    generator, steps = LSTModel.get_generator(X, y, 2, flip, num_shifts=num_shifts)
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

    assert (np.tile(y, (num_times)) == new_y).all()
    _check_new_X(X, new_X, num_times, flip)
