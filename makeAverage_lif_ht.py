import glob
import os
import shutil

import numpy as np
from readlif.reader import LifFile
from tifffile import imsave

path = r'D:\Projects\Denoising-STED\20220913-RPI\tubulin-clathrine-highthroughput'
file_name = r'D:\Projects\Denoising-STED\20220913-RPI\tubulin-clathrine-highthroughput\TubulinClatherin high throughput.lif'

os.mkdir(os.path.join(path, '', 'low SNR'))
os.mkdir(os.path.join(path, '', 'GT'))
path_1frame = os.path.join(path, '', 'low SNR', '')
path_avg = os.path.join(path, '', 'GT', '')

# os.chdir(path)
# fil=glob.glob("*.lif")
# lf = len(fil)


new = LifFile(file_name)
for i in range(1):
    i = 2
    new_img = new.get_image(i)
    x = new_img.dims[0]
    y = new_img.dims[1]
    for p in range(new_img.dims[4]):
        img_avg = np.zeros((x, y))

        img_avg = new_img.get_frame(z=0, t=0, c=1, m=p)+img_avg

        img_avg = img_avg / img_avg.max()
        img_avg = np.uint16(img_avg * (2 ** 16 - 1))
        imsave(path_avg + str(p) + '.tif', img_avg)
