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


def log_mae_on_type(orig, preds):
    """
    2nd column of orig has type.
    For each type we compute the log of mean of absolute error.
    We return the mean of lmae  of all types
    """
    orig = tf.reshape(orig, (-1, 2))
    preds = tf.reshape(preds, (-1, 1))
    y = orig[:, 0]
    edge_types = orig[:, 1]

    log_mae_each_type = 0
    typ_count = 0

    get_zero = lambda: tf.constant(0.)
    get_one = lambda: tf.constant(1.)

    for typ in list(range(8)):
        mask = tf.equal(edge_types, typ)
        y_typ = tf.boolean_mask(y, mask)
        preds_typ = tf.boolean_mask(preds, mask)
        lmae = log_mae(y_typ, preds_typ)
        is_empty = tf.equal(tf.size(y_typ), 0)
        get_lmae = lambda: lmae
        log_mae_each_type += tf.case([(is_empty, get_zero)], default=get_lmae)
        typ_count += tf.case([(is_empty, get_zero)], default=get_one)

    return log_mae_each_type / typ_count


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
