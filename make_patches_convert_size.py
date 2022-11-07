import os, glob
from tifffile import imread, imwrite
import numpy as np

file_path = r'C:\Users\va332845\OneDrive - Knights - University of Central Florida\Nanoscopy Group\Experiments\STED\Collaboration with Christian Eggeling\221018 Vesicle dwell time\m4'
save_path = r'C:\Users\va332845\OneDrive - Knights - University of Central Florida\Nanoscopy Group\Experiments\STED\Collaboration with Christian Eggeling\221018 Vesicle dwell time\m4\patches'

list_files = glob.glob(os.path.join(file_path, '', '*.tif'))

patch_size = 256
threshold = 0.0
count1 = 0
for i in list_files:
    f = imread(i).astype(np.float64)
    f = f / f.max()
    f = f - f.min()
    f = f / f.max()
    f[f < 0] = 0
    w = f.shape[-1]
    h = f.shape[-2]
    if len(f.shape) == 3:
        t = f.shape[0]
    else:
        t = 1
        f = f.reshape((1, f.shape[0], f.shape[1]))
    w1 = int(np.floor((patch_size - w) / 2))
    h1 = int(np.floor((patch_size - h) / 2))
    x = np.zeros((t, patch_size, patch_size))
    x[:, h1:h1 + h, w1:w1 + w] = f
    max_x = x.max(axis=(1, 2))
    max_x[max_x == 0] = 1
    max_x = max_x.reshape((t, 1, 1))
    x = x / max_x
    y = np.uint16(x * (2 ** 16 - 1))
    imwrite(os.path.join(save_path, '', str(count1) + '.tif'), y, imagej=True, metadata={'axes': 'TYX'})
    count1 = count1 + 1
