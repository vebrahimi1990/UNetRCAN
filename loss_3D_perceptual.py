import numpy as np
import tensorflow as tf

from keras.models import Input, Model, load_model

patch_size = 160
mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                  input_tensor=Input(shape=(patch_size, patch_size, 3)))

inter_vgg = []
LL = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17]
for i in LL:
    inter_vgg.append(Model(inputs=vgg.input, outputs=vgg.get_layer(vgg.layers[i].name).output))


def norm_mse_loss(prediction, gt):
    n_mse = mse(prediction, gt)
    norm_mse = tf.squeeze(n_mse)
    return norm_mse


# def fft_loss(prediction, gt):
#     prediction = tf.transpose(prediction, perm=[0, 3, 1, 2])
#     gt = tf.transpose(gt, perm=[0, 3, 1, 2])
#
#     fft_prediction = tf.signal.fftshift(tf.signal.rfft2d(prediction))
#     fft_gt = tf.signal.fftshift(tf.signal.rfft2d(gt))
#
#     fft_prediction = tf.transpose(fft_prediction, perm=[0, 2, 3, 1])
#     fft_gt = tf.transpose(fft_gt, perm=[0, 2, 3, 1])
#
#     loss = norm_mse_loss(fft_prediction, fft_gt)
#     loss = tf.cast(loss, tf.float32)
#     return loss


def ssim_loss(prediction, gt):
    prediction = tf.math.reduce_max(prediction, axis=3)
    gt = tf.math.reduce_max(gt, axis=3)
    loss = 1.0 - tf.math.reduce_mean(tf.image.ssim(prediction, gt, max_val=1))
    return loss


def perceptual_loss(prediction, gt):
    prediction = tf.math.reduce_max(prediction, axis=3)
    gt = tf.math.reduce_max(gt, axis=3)
    loss = 0
    prediction = tf.image.grayscale_to_rgb(prediction)
    gt = tf.image.grayscale_to_rgb(gt)
    for m in range(len(LL)):
        vgg_prediction = inter_vgg[m](prediction)
        vgg_gt = inter_vgg[m](gt)
        loss = loss + norm_mse_loss(vgg_prediction, vgg_gt)
    return loss


# def gauss(x, mean, sig):
#     f = tf.math.exp(-((x - mean) ** 2) / (0.6 * sig ** 2))
#     return f
#
#
# m = tf.linspace(0, 1, 100)
# m = tf.cast(m, dtype=tf.float32)
# s = 0.005
#
#
# def hist(prediction, gt):
#     loss = 0
#     prediction = tf.math.reduce_max(prediction, axis=3)
#     gt = tf.math.reduce_max(gt, axis=3)
#     for i in range(len(m)):
#         hist_pred = tf.math.reduce_sum(gauss(prediction, m[i], s), axis=(1, 2))
#         hist_gt = tf.math.reduce_sum(gauss(gt, m[i], s), axis=(1, 2))
#         loss = loss + norm_mse_loss(hist_pred, hist_gt)
#     return loss


def generator_loss(prediction, gt):
    prediction1, prediction2 = prediction
    norm_mse1 = norm_mse_loss(prediction1, gt)
    percept_loss1 = perceptual_loss(prediction1, gt)
    norm_mse2 = norm_mse_loss(prediction2, gt)
    sim_loss2 = ssim_loss(prediction2, gt)
    gen_loss1 = norm_mse1 + 0.01 * percept_loss1
    gen_loss2 = norm_mse2 + sim_loss2
    total_gen_loss = gen_loss1 + gen_loss2
    return total_gen_loss

# def ch_loss(pred, gt):
#     norm = tf.norm(tf.norm(pred - gt, axis=(1, 2)), axis=1)
#     norm = tf.squeeze(norm)
#     norm = tf.pow(norm, 2)
#     norm = norm / (256 * 256) + 1e-6
#     norm = tf.pow(norm, 0.5)
#     c_loss = tf.math.reduce_mean(norm)
#     return c_loss
#
#
# def edge_loss(pred, gt):
#     pred = tf.math.reduce_max(pred, axis=3)
#     gt = tf.math.reduce_max(gt, axis=3)
#     kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
#     kernel = kernel.reshape((3, 3, 1, 1))
#     pred = tf.nn.conv2d(pred, kernel, strides=[1, 1, 1, 1], padding='VALID')
#     gt = tf.nn.conv2d(gt, kernel, strides=[1, 1, 1, 1], padding='VALID')
#     e_loss = ch_loss(pred, gt)
#     return e_loss


# def generator_loss(prediction, gt):
#     c_loss = ch_loss(prediction, gt)
#     e_loss = edge_loss(prediction, gt)
#     gen_loss = c_loss + 0.5 * e_loss
#     return gen_loss
