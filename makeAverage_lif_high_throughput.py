import glob
import os
import shutil

import numpy as np
from readlif.reader import LifFile
from tifffile import imsave

path = r'D:\Projects\Denoising-STED\20220913-RPI\tubulin-clathrine-highthroughput\clathrine'
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


for k in range(c1):
    img_avg = np.zeros((x, y))
    img_avg = new_img.get_frame(z=0, t=0, c=0, m=k) + img_avg
    img_avg = img_avg/img_avg.max()
    img_avg = np.uint16(img_avg * (2 ** 16 - 1))
    imsave(os.path.join(path, '') + 'image_' + str(k) + '.tif', img_avg.squeeze())
