import trimesh
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'out_problem.stl'
mesh = trimesh.load(path, force='mesh')
faces = len(mesh.faces) if hasattr(mesh, 'faces') else None
watertight = getattr(mesh, 'is_watertight', None)
euler = getattr(mesh, 'euler_number', None)

# count open edges (edges referenced by only one face)
edge_count = {}
for f in mesh.faces:
    for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
        e = tuple(sorted((int(a), int(b))))
        edge_count[e] = edge_count.get(e, 0) + 1
open_edges = sum(1 for v in edge_count.values() if v == 1)

print('file=', path)
print('watertight=', watertight)
print('faces=', faces)
print('euler_number=', euler)
print('open_edges=', open_edges)
print('bounds=', mesh.bounds.tolist())
print('vertices=', len(mesh.vertices))
print('components=', len(mesh.split(only_watertight=False)))
