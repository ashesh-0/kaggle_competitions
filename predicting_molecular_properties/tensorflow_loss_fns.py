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
