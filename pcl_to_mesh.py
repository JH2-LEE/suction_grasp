import numpy as np

import open3d as o3d


# import open3d


# load pcl
points = np.load("pcl1.npy")
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(points)

# o3d.visualization.draw_geometries([pcl])

alpha = 0.002
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcl)

# o3d.visualization.draw_geometries([tetra_mesh], mesh_show_back_face=True)

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcl, alpha, tetra_mesh, pt_map
)

tri_mesh = o3d.geometry.TetraMesh.extract_triangle_mesh(pcl.points, 1)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([tri_mesh], mesh_show_back_face=True)
