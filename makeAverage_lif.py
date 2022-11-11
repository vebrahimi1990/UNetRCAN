import glob
import os
import shutil

import numpy as np
from readlif.reader import LifFile
from tifffile import imsave

path = r'D:\Projects\Denoising-STED\20220913-RPI\STED power dependence\tom20'
file_name = r'D:\Projects\Denoising-STED\20220913-RPI\tom20-crossmod-0-10-20-40-80.lif'

os.mkdir(os.path.join(path, '', 'low SNR'))
os.mkdir(os.path.join(path, '', 'GT'))
path_1frame = os.path.join(path, '', 'low SNR', '')
path_avg = os.path.join(path, '', 'GT', '')

# os.chdir(path)
# fil=glob.glob("*.lif")
# lf = len(fil)


new = LifFile(file_name)
for i in range(new.num_images):
    new_img = new.get_image(i)
    c = new_img.channels
    # c = 1
    x = new_img.dims[0]
    y = new_img.dims[1]
    z = new_img.dims[2]
    t = new_img.dims[3]
    # c = new_img.dims[4]
    img_avg = np.zeros((c, z, x, y))
    img_frame = np.zeros((c, z, x, y))

    if t > 6:
        for j in range(c):
            for k in range(z):
                for m in range(3):
                    # m = m + 3
                    img_frame[j, k, :, :] = new_img.get_frame(z=k, t=m, c=j) + img_frame[j, k, :, :]
            img_frame[j, :, :, :] = img_frame[j, :, :, :] / img_frame[j, :, :, :].max()
        img_frame = np.uint16(img_frame * (2 ** 16 - 1))
        imsave(path_1frame + str(i) + '.tif', img_frame.squeeze(), imagej=True, metadata={'axes': 'CYX'})
        # imsave(path_1frame + str(i) + '.tif', img_frame.squeeze())
    if t == 1:
        for j in range(c):
            for k in range(z):
                for m in range(t):
                    img_avg[j, k, :, :] = new_img.get_frame(z=k, t=m, c=j) + img_avg[j, k, :, :]
            img_avg[j, :, :, :] = img_avg[j, :, :, :] / img_avg[j, :, :, :].max()
        img_avg = np.uint16(img_avg * (2 ** 16 - 1))
        # imsave(path_avg + str(i) + '.tif', img_avg.squeeze(), imagej=True, metadata={'axes': 'CYX'})
        imsave(path_avg + str(i) + '.tif', img_avg.squeeze())
