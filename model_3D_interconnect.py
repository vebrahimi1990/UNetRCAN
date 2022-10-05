import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.activations import sigmoid
from keras.layers import Dropout, LeakyReLU, ReLU, UpSampling3D
from keras.layers.convolutional import Conv3D
from keras.layers.merge import concatenate, add, multiply
from keras.layers.pooling import MaxPooling3D, GlobalAveragePooling3D
from keras.models import Model


# w_init = 'glorot_uniform'


def kinit(size, filters):
    n = 1 / np.sqrt(size * size * filters)
    w_init = tf.keras.initializers.RandomUniform(minval=-n, maxval=n)
    #     w_init='random_normal'
    return w_init


def conv_block(inputs, filters, kernel):
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    y = Conv3D(filters=filters, kernel_size=3, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def CAB(inputs, filters_cab, filters, kernel):
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    z = GlobalAveragePooling3D(data_format='channels_last', keepdims=True)(x)
    z = Conv3D(filters=filters_cab, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(z)
    z = LeakyReLU()(z)
    z = Conv3D(filters=filters, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(z)
    z = sigmoid(z)
    z = multiply([z, x])
    z = add([z, inputs])
    return z


def RG(inputs, num_CAB, filters, filters_cab, kernel, dropout):
    x = inputs
    for i in range(num_CAB):
        x = CAB(x, filters_cab, filters, kernel)
        x = Dropout(dropout)(x)
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    x = add([x, inputs])
    return x


def RiR(inputs, num_RG, num_RCAB, filters, filters_cab, kernel, en_out, de_out, dropout):
    x = inputs
    for i in range(num_RG):
        x = RG(x, num_RCAB, filters, filters_cab, kernel, dropout)
        # x = add([x, en_out[i]])
        # x = add([x, de_out[i]])
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    x = add([x, inputs])
    return x


def make_RCAN(inputs, filters, filters_cab, num_RG, num_RCAB, kernel, en_out, de_out, dropout):
    x = Conv3D(filters=filters, kernel_size=kernel, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(inputs)
    x = RiR(x, num_RG, num_RCAB, filters, filters_cab, kernel, en_out, de_out, dropout)
    x = Dropout(dropout)(x)
    x = Conv3D(filters=1, kernel_size=1, kernel_initializer=kinit(kernel, filters),
               bias_initializer=kinit(kernel, filters), padding="same")(x)
    return x


def make_generator(inputs, filters, num_filters, filters_cab, num_RG, num_RCAB, kernel_shape, dropout):
    skip_x = []
    skip_y = []
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(x, f, kernel_shape)
        x = Dropout(dropout)(x)
        skip_x.append(x)
        x = MaxPooling3D((2, 2, 2))(x)

    x = conv_block(x, 2 * filters[-1], kernel_shape)
    skip_x.append(x)
    x = conv_block(x, 2 * filters[-1], kernel_shape)
    skip_y.append(x)
    filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(filters):
        x = UpSampling3D(size=(2, 2, 2), data_format='channels_last')(x)
        xs = skip_x[i + 1]
        xs = CAB(xs, filters_cab=4, filters=f, kernel=3)
        x = concatenate([x, xs])
        x = conv_block(x, f, kernel_shape)
        skip_y.append(x)
        x = Dropout(dropout)(x)

    x = Conv3D(filters=1, kernel_size=1, kernel_initializer=kinit(3, filters[0]), bias_initializer=kinit(3, 1),
               padding="same")(x)
    y = concatenate([x, inputs])
    skip_x.reverse()
    skip_y.reverse()

    # for i in range(len(filters) + 1):
    #     skip_x[i] = UpSampling3D(size=2 ** i, data_format='channels_last')(skip_x[i])
    #     skip_x[i] = Conv3D(filters=num_filters, kernel_size=1, kernel_initializer=kinit(1, num_filters))(skip_x[i])
    #     skip_y[i] = UpSampling3D(size=2 ** i, data_format='channels_last')(skip_y[i])
    #     skip_y[i] = Conv3D(filters=num_filters, kernel_size=1, kernel_initializer=kinit(1, num_filters))(skip_y[i])
    y = make_RCAN(inputs=y, filters=num_filters, filters_cab=filters_cab, num_RG=num_RG, num_RCAB=num_RCAB,
                  kernel=kernel_shape, en_out=skip_x, de_out=skip_y, dropout=dropout)
    model = Model(inputs=[inputs], outputs=[x, y])
    return model
