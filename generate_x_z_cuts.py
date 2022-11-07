import os
from tifffile import imread, imwrite
import numpy as np

file_path = r'C:\Users\va332845\OneDrive - Knights - University of Central Florida\Nanoscopy Group\Experiments\STED\Collaboration with Christian Eggeling\221011 Vesicle dwell time\deconv\results'
save_path = r'C:\Users\va332845\OneDrive - Knights - University of Central Florida\Nanoscopy Group\Experiments\STED\Collaboration with Christian Eggeling\221011 Vesicle dwell time\deconv\results1'

list_files = os.listdir(file_path)
M = len(list_files)
count1 = 0
for i in list_files:
    f = imread(os.path.join(file_path, '', i)).astype(np.float64)
    f = f / f.max()
    f = f - f.min()
    f = f / f.max()
    f[f < 0] = 0
    nx = f.shape[-1]
    nz = f.shape[-2]
    f = f.reshape((f.shape[0], f.shape[1], f.shape[3]))
    f = np.uint16(f * (2 ** 16 - 1))
    imwrite(os.path.join(save_path, '', str(count1) + '.tif'), f, imagej=True, metadata={'axes': 'TYX'})
    count1 = count1 + 1
