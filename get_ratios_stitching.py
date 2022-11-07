import glob
import os
import shutil

import numpy as np
from readlif.reader import LifFile
from tifffile import imsave

path = r'D:\Projects\Denoising-STED\20220913-RPI\tubulin-clathrine-highthroughput'
file_name = r'D:\Projects\Denoising-STED\20220913-RPI\tubulin-clathrine-highthroughput\TubulinClatherin high throughput.lif'

new = LifFile(file_name)
# for i in range(new.num_images):
i = 0
new_img = new.get_image(i)
c = new_img.channels
# c = 1
x = new_img.dims[0]
y = new_img.dims[1]
z = new_img.dims[2]
t = new_img.dims[3]
c1 = new_img.dims[4]
img_avg = np.zeros((c, x, y))
img_ratio = np.zeros((c, c1))

for k in range(c1):
    for j in range(c):
        img_avg[j, :, :] = new_img.get_frame(z=0, t=0, c=j, m=k) + img_avg[j, :, :]
    img_ratio[:, k] = np.mean(img_avg, axis=(1, 2))

path_save = r'D:\Projects\Denoising-STED\20220913-RPI\tubulin-clathrine-highthroughput\ratios.csv'

np.savetxt(path_save, np.transpose(img_ratio), delimiter=",")
