import base64
import io
from PIL import Image
import numpy as np
import sys
import traceback
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import generate_stl_via_shapely, generate_stl_via_openscad, generate_stl_marching_cubes

path = sys.argv[1] if len(sys.argv) > 1 else 'test_images/problem.png'
img = Image.open(path).convert('L')
width, height = img.size
arr = np.array(img)
mask_array = arr < 128
mask_array = mask_array.astype(np.uint8)

print('image size', width, height)

# shapely
try:
    s = generate_stl_via_shapely(mask_array, width, height, z_offset=0, thickness=5.0, target_width_mm=50.0)
    if s:
        with open('out_shapely_direct.stl', 'wb') as f:
            f.write(s.encode('utf-8'))
        print('shapely produced output')
    else:
        print('shapely returned None')
except Exception as e:
    print('shapely exception')
    traceback.print_exc()

# openscad
try:
    s = generate_stl_via_openscad(mask_array, width, height, z_offset=0, thickness=5.0, target_width_mm=50.0)
    if s:
        with open('out_openscad_direct.stl', 'wb') as f:
            f.write(s.encode('utf-8'))
        print('openscad produced output')
    else:
        print('openscad returned None')
except Exception as e:
    print('openscad exception')
    traceback.print_exc()

# marching cubes
try:
    s = generate_stl_marching_cubes(mask_array, width, height, z_offset=0, thickness=5.0, max_dim=1024)
    if s:
        with open('out_mc_direct.stl', 'wb') as f:
            f.write(s.encode('utf-8'))
        print('marching cubes produced output')
    else:
        print('marching cubes returned None')
except Exception as e:
    print('marching cubes exception')
    traceback.print_exc()

print('done')
