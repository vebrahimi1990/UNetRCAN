import random

import numpy as np
from skimage.io import imread
from tifffile import imread
from skimage import exposure
from skimage.exposure import match_histograms


def data_generator(GT_image_dr, lowSNR_image_dr, patch_size, n_patches, n_channel, threshold, ratio, lp, augment=False,
                   shuffle=False, add_noise=False):
    gt = imread(GT_image_dr)
    low = imread(lowSNR_image_dr)
    if len(gt.shape) == 3:
        gt = np.reshape(gt, (gt.shape[0], 1, gt.shape[1], gt.shape[2]))
        low = np.reshape(low, (low.shape[0], 1, low.shape[1], low.shape[2]))
    if len(gt.shape) == 2:
        gt = np.reshape(gt, (1, 1, gt.shape[0], gt.shape[1]))
        low = np.reshape(low, (1, 1, low.shape[0], low.shape[1]))
    gt = gt.astype(np.float64)
    gt = gt / gt.max()
    print(gt.shape)
    low = low.astype(np.float64)
    low = low / low.max()
    m = gt.shape[0]
    # m = 5
    img_size = gt.shape[2]

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    y = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)

    rr = np.floor(np.linspace(0, img_size - patch_size, n_patches))
    rr = rr.astype(np.int32)
    cc = np.floor(np.linspace(0, gt.shape[3] - patch_size, n_patches))
    cc = cc.astype(np.int32)
    # for i in range(len(low)):
    #     low[i] = match_histograms(low[i], gt[i])

    count = 0
    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, 0] = low[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                y[count, :, :, 0] = gt[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                count = count + 1

    if add_noise:
        for i in range(len(x)):
            # x[i] = np.random.poisson(y[i]**0.5, size=y[i].shape)
            # x[i] = y[i] + np.random.normal(loc=0.01 + 0.005 * (np.random.rand(1)[0] - 0.5),
            #                                scale=0.3 + 0.01 * (np.random.rand(1)[0] - 0.5), size=y[i].shape)
            x[i] = np.random.poisson(y[i] / lp, size=y[i].shape)

    x[x < 0] = 0
    # x = x/x.max()
    for i in range(len(x)):
        if x[i].max() > 0:
            x[i] = x[i] / x[i].max()
            y[i] = y[i] / y[i].max()

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        yy = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)

        xx[0:count, :, :, :] = x
        xx[count:2 * count, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :] = np.flip(x, axis=(1, 2))

        yy[0:count, :, :, :] = y
        yy[count:2 * count, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        yy = y

    norm_x = np.linalg.norm(xx, axis=(1, 2))
    # norm_x = norm_x-15
    ind_norm = np.where(norm_x > threshold)[0]
    print(len(ind_norm))

    xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))
    yyy = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))

    for i in range(len(ind_norm)):
        xxx[i] = xx[ind_norm[i]]
        yyy[i] = yy[ind_norm[i]]

    aa = np.linspace(0, len(xxx) - 1, len(xxx))
    random.shuffle(aa)
    aa = aa.astype(int)

    xxs = np.empty(xxx.shape, dtype=np.float64)
    yys = np.empty(yyy.shape, dtype=np.float64)

    if shuffle:
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        yys = yyy

    #     for i in range(len(xxs)):
    #         xxs[i] = np.random.poisson(yys[i]/(0.005+0.0025*(np.random.rand(1)[0]-0.5)))
    #         xxs[i] = np.random.normal(xxs[i]/(0.005+0.0025*(np.random.rand(1)[0]-0.5)))
    #     lp = 0.01
    #     for i in range(len(xxs)):
    #         xxs[i] = np.random.poisson(xxs[i]/(lp+lp/2*(np.random.rand(1)[0]-0.5)))
    # #         xxs[i] = np.random.normal(xxs[i]/(0.01+0.005*(np.random.rand(1)[0]-0.5)))

    xxs[xxs < 0] = 0
    for i in range(len(xxs)):
        for j in range(n_channel):
            xxs[i, :, :, j] = xxs[i, :, :, j] / xxs[i, :, :, j].max()
            if yys[i, :, :, j].max() > 0:
                yys[i, :, :, j] = yys[i, :, :, j] / yys[i, :, :, j].max()

    # Split train and valid
    ratio = ratio
    m1 = np.floor(xxs.shape[0] * ratio).astype(np.int32)
    x_train = xxs[0:m1]
    y_train = yys[0:m1]
    # x_train = xxs
    # y_train = yys
    x_valid = xxs[m1::]
    y_valid = yys[m1::]

    # x_train = x_train.reshape(x_train.shape+(1,))
    # y_train = y_train.reshape(y_train.shape+(1,))
    # x_valid = x_valid.reshape(x_valid.shape+(1,))
    # y_valid = y_valid.reshape(y_valid.shape+(1,))

    print('The training set shape is:', x_train.shape)
    print('The validation set shape is:', x_valid.shape)
    return x_train, y_train, x_valid, y_valid
