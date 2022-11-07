import numpy as np
from tifffile import imread, imwrite

path = r'D:\Projects\Denosing-STED-Abberior\2022-11-03_UCF_DemoSamples\3D STED_TOM20\test\prediction'

nx = np.linspace(0, 2048-256, 8).astype(np.int64)
ny = np.linspace(0, 2048-256, 8).astype(np.int64)
print(nx)
