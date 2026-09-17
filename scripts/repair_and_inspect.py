import trimesh
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'out_shapely_direct.stl'
mesh = trimesh.load(path, force='mesh')
print('initial watertight=', mesh.is_watertight)

try:
    print('remove duplicate faces')
    mesh.remove_duplicate_faces()
except Exception as e:
    print('dup faces error', e)

try:
    print('remove degenerate')
    mesh.remove_degenerate_faces()
except Exception as e:
    print('degenerate error', e)

try:
    print('merge vertices')
    mesh.merge_vertices()
except Exception as e:
    print('merge error', e)

try:
    print('fix normals')
    trimesh.repair.fix_normals(mesh)
except Exception as e:
    print('fix_normals error', e)

try:
    print('fill holes')
    trimesh.repair.fill_holes(mesh)
except Exception as e:
    print('fill_holes error', e)

try:
    print('fix winding')
    trimesh.repair.fix_winding(mesh)
except Exception as e:
    print('fix_winding error', e)

try:
    mesh.remove_unreferenced_vertices()
except Exception as e:
    print('remove_unreferenced_vertices error', e)

print('final watertight=', mesh.is_watertight)
print('euler=', getattr(mesh, 'euler_number', None))
print('faces=', len(mesh.faces))
mesh.export('out_shapely_repaired.stl', file_type='stl_ascii')
print('wrote out_shapely_repaired.stl')
