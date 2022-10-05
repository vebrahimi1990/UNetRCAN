import glob
import os
import shutil

import numpy as np
from readlif.reader import LifFile
from readlif.reader import
from tifffile import imsave

file_name = r'D:\Projects\Denoising-STED\20220913-RPI\TubulinClatherin high throughput.lifext'

new = LifFile(file_name)