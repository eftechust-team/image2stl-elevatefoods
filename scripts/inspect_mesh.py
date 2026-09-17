import sys
import trimesh

path = sys.argv[1] if len(sys.argv) > 1 else 'out_shapely_direct.stl'
mesh = trimesh.load(path, force='mesh')
print('file=', path)
print('is_watertight=', mesh.is_watertight)
print('is_winding_consistent=', getattr(mesh, 'is_winding_consistent', None))
print('is_volume=', getattr(mesh, 'is_volume', None))
print('euler_number=', getattr(mesh, 'euler_number', None))
print('bounds=', mesh.bounds.tolist())
print('vertices=', len(mesh.vertices), 'faces=', len(mesh.faces))
print('is_watertight_fast=', mesh.bounds is not None)
# non-manifold edges/vertices
print('face_adjacency_edges=', len(mesh.face_adjacency_edges))
non_manifold_edges = sum(1 for e,count in zip(mesh.edges_unique, mesh.edges_unique_counts) if count>2)
print('non_manifold_edges_count=', non_manifold_edges)
try:
    print('contains_infinite=', mesh.is_empty)
except Exception:
    pass
