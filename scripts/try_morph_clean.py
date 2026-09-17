import numpy as np
from PIL import Image
from skimage import morphology
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import generate_stl_marching_cubes

img = Image.open(sys.argv[1] if len(sys.argv)>1 else 'test_images/problem.png').convert('L')
arr = np.array(img)
mask = arr < 128
width, height = img.size

params = []
for closing in (1,3,5):
    for min_hole in (0,50,200,1000):
        params.append((closing, min_hole))

results = []
for closing, min_hole in params:
    print('trying closing', closing, 'min_hole', min_hole)
    m = mask.copy()
    if closing>0:
        se = morphology.disk(closing)
        m = morphology.binary_closing(m, footprint=se)
    if min_hole>0:
        m = morphology.remove_small_holes(m, area_threshold=min_hole)
    s = generate_stl_marching_cubes(m, width, height, z_offset=0, thickness=5.0, max_dim=512)
    fname = f'out_mc_closing{closing}_hole{min_hole}.stl'
    if s:
        open(fname,'wb').write(s.encode('utf-8'))
        print('wrote', fname)
    else:
        print('no output')

print('done')
