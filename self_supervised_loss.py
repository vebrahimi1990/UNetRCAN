import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')


def norm_mse_loss(prediction, gt):
    n_mse = mse(prediction, gt)
    norm_mse = tf.squeeze(n_mse)
    return norm_mse


def poisson_noise(prediction, lamb):
    out = tf.random.poisson(shape=prediction.shape, lam=lamb * prediction)
    return out


def loss(lamb, prediction, noisy):
    pred_noisy = poisson_noise(prediction, lamb)
    value_loss = norm_mse_loss(pred_noisy, noisy)
    return value_loss
