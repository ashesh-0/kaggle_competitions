# Taken from  https://www.kaggle.com/fnands/1-mpnn
import tensorflow as tf
import numpy as np


def mse(orig, preds):

    # Mask values for which no scalar coupling exists
    mask = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums = tf.boolean_mask(orig, mask)
    preds = tf.boolean_mask(preds, mask)

    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(nums, preds)))

    return reconstruction_error


def log_mse(orig, preds):
    reconstruction_error = tf.math.log(mse(orig, preds))
    return reconstruction_error


def mse_on_type(orig, preds):
    return _metric_on_type(orig, preds, mse)


def log_mae_on_type(orig, preds):
    return _metric_on_type(orig, preds, log_mae)


def _metric_on_type(orig, preds, metric_fn):
    """
    2nd column of orig has type.
    For each type we compute the log of mean of absolute error.
    We return the mean of metric  of all types
    """
    orig = tf.reshape(orig, (-1, 2))
    preds = tf.reshape(preds, (-1, 1))

    y = orig[:, 0]
    edge_types = orig[:, 1]

    metric_each_type = 0
    typ_count = 0

    get_zero = lambda: tf.constant(0.)
    get_one = lambda: tf.constant(1.)

    for typ in list(range(8)):
        mask = tf.equal(edge_types, typ)
        y_typ = tf.boolean_mask(y, mask)
        preds_typ = tf.boolean_mask(preds, mask)
        metric = metric_fn(y_typ, preds_typ)
        is_empty = tf.equal(tf.size(y_typ), 0)
        get_metric = lambda: metric
        metric_each_type += tf.case([(is_empty, get_zero)], default=get_metric)
        typ_count += tf.case([(is_empty, get_zero)], default=get_one)

    avg_metric = lambda: metric_each_type / typ_count
    return tf.case([(tf.equal(typ_count, 0), get_zero)], default=avg_metric)


def mae(orig, preds):

    # Mask values for which no scalar coupling exists
    mask = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums = tf.boolean_mask(orig, mask)
    preds = tf.boolean_mask(preds, mask)

    reconstruction_error = tf.reduce_mean(tf.abs(tf.subtract(nums, preds)))

    return reconstruction_error


def log_mae(orig, preds):
    reconstruction_error = tf.math.log(mae(orig, preds))
    return reconstruction_error


def step_decay(epoch, learning_rate):
    initial_lrate = learning_rate
    drop = 0.1
    epochs_drop = 20.0
    lrate = initial_lrate * np.power(drop, np.floor((epoch) / epochs_drop))
    tf.print("Learning rate: ", lrate)
    return lrate
