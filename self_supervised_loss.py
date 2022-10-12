import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')


def norm_mse_loss(prediction, gt):
    n_mse = mse(prediction, gt)
    norm_mse = tf.squeeze(n_mse)
    return norm_mse


def loss(prediction, noisy):
    value_loss = norm_mse_loss(prediction, noisy)
    return value_loss
