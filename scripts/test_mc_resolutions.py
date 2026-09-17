import numpy as np
from PIL import Image
import sys
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import generate_stl_marching_cubes

img = Image.open(sys.argv[1] if len(sys.argv)>1 else 'test_images/problem.png').convert('L')
arr = np.array(img)
mask = arr < 128
width, height = img.size

for d in (256, 512, 1024):
    print('trying max_dim', d)
    s = generate_stl_marching_cubes(mask, width, height, z_offset=0, thickness=5.0, max_dim=d)
    fn = f'out_mc_{d}.stl'
    if s:
        open(fn,'wb').write(s.encode('utf-8'))
        print('wrote', fn)
    else:
        print('no output for', d)
