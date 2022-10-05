import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.activations import sigmoid
from keras.layers import Dropout, LeakyReLU, ReLU, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate, add, multiply
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


# w_init = 'glorot_uniform'


def kinit(size, filters):
    n = 1 / np.sqrt(size * size * filters)
    w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    # w_init = 'random_normal'
    # w_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
    return w_init


def kinit_bias(size, filters):
    # n = 1 / np.sqrt(size * size * filters)
    # w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    # w_init = 'random_normal'
    w_init = 'zeros'
    return w_init


def conv_block(inputs, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(x)
    y = Conv2D(filters=filters, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(inputs)
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def CAB(inputs, filters_cab, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(x)
    z = GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(x)
    z = Conv2D(filters=filters_cab, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(z)
    z = LeakyReLU()(z)
    z = Conv2D(filters=filters, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(z)
    z = sigmoid(z)
    z = multiply([z, x])
    z = add([z, inputs])
    return z


def RG(inputs, num_CAB, filters, filters_cab, kernel, dropout):
    x = inputs
    for i in range(num_CAB):
        x = CAB(x, filters_cab, filters, kernel)
        x = Dropout(dropout)(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(x)
    x = add([x, inputs])
    return x


def RiR(inputs, num_RG, num_RCAB, filters, filters_cab, kernel, en_out, de_out, dropout):
    x = inputs
    for i in range(num_RG):
        x = RG(x, num_RCAB, filters, filters_cab, kernel, dropout)
        x = add([x, en_out[i]])
        x = add([x, de_out[i]])
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(x)
    x = add([x, inputs])
    return x


def make_RCAN(inputs, filters, filters_cab, num_RG, num_RCAB, kernel, en_out, de_out, dropout):
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(inputs)
    x = RiR(x, num_RG, num_RCAB, filters, filters_cab, kernel, en_out, de_out, dropout)
    x = Dropout(dropout)(x)
    x = Conv2D(filters=1, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit_bias(kernel, filters), padding="same")(x)
    return x


def make_decoder(inputs, filters, en_out, kernel_shape, dropout):
    skip_x = en_out
    x = inputs
    for i, f in enumerate(filters):
        x = UpSampling2D(size=2, data_format='channels_last')(x)
        xs = skip_x[i + 1]
        x = concatenate([x, xs])
        x = conv_block(x, f, kernel_shape)
        x = Dropout(dropout)(x)
    return x


def make_generator(inputs, filters, num_gaussian, kernel_shape, dropout):
    skip_x = []
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(x, f, kernel_shape)
        x = Dropout(dropout)(x)
        skip_x.append(x)
        x = MaxPooling2D(2)(x)

    x = conv_block(x, 2 * filters[-1], kernel_shape)
    skip_x.append(x)
    x = conv_block(x, 2 * filters[-1], kernel_shape)
    filters.reverse()
    skip_x.reverse()
    amp = make_decoder(x, filters, skip_x, kernel_shape, dropout)
    mx = make_decoder(x, filters, skip_x, kernel_shape, dropout)
    sx = make_decoder(x, filters, skip_x, kernel_shape, dropout)
    my = make_decoder(x, filters, skip_x, kernel_shape, dropout)
    sy = make_decoder(x, filters, skip_x, kernel_shape, dropout)
    mz = make_decoder(x, filters, skip_x, kernel_shape, dropout)
    sz = make_decoder(x, filters, skip_x, kernel_shape, dropout)

    amp = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                 bias_initializer=kinit(3, 1),
                 padding="same")(amp)
    mx = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                bias_initializer=kinit(3, 1),
                padding="same")(mx)
    sx = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                bias_initializer=kinit(3, 1),
                padding="same")(sx)
    my = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                bias_initializer=kinit(3, 1),
                padding="same")(my)
    sy = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                bias_initializer=kinit(3, 1),
                padding="same")(sy)
    mz = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                bias_initializer=kinit(3, 1),
                padding="same")(mz)
    sz = Conv2D(filters=num_gaussian, kernel_size=1, kernel_initializer=kinit(3, filters[0]),
                bias_initializer=kinit(3, 1),
                padding="same")(sz)
    model = Model(inputs=[inputs], outputs=[amp, mx, sx, my, sy, mz, sz])
    return model
