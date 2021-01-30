import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

# pcd = o3d.io.read_point_cloud("pcl1.npy", format='xyz')

points = np.load("pcl_test/pcl1.npy")
print(points.shape)
print("np load")
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(points)

# estimate normal
pcl.normals = o3d.utility.Vector3dVector(
    np.zeros((1, 3))
)  # invalidate existing normals
pcl.estimate_normals()
o3d.visualization.draw_geometries([pcl], point_show_normal=True)
print("pcl")
# voxel grid
"""
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, voxel_size=0.004)
o3d.visualization.draw_geometries([voxel_grid])
"""

# alpha shape in surface reconstruction
"""
alpha = 0.5
alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha)
alpha_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True)
"""
"""
alpha = 0.3
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcl)
alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    pcl, alpha, tetra_mesh, pt_map
)
alpha_mesh.compute_convex_hull()
alpha_mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([alpha_mesh], mesh_show_back_face=True)
"""

# ball pivoting
"""
radii = [0.005, 0.01, 0.02, 0.04]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcl, o3d.utility.DoubleVector(radii)
)
# o3d.visualization.draw_geometries([pcl, rec_mesh])
o3d.visualization.draw_geometries([pcl, rec_mesh], mesh_show_back_face=True)
"""

distances = pcl.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist
radius = 1.2 * avg_dist
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcl, o3d.utility.DoubleVector([radius, radius * 2])
)
print(mesh)
# o3d.visualization.draw_geometries([pcl, mesh])
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
print(np.asarray(mesh.vertices))
print(np.asarray(mesh.triangles))

# poisson
"""
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcl, depth=9
    )
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
densities = np.asarray(densities)
density_colors = plt.get_cmap("plasma")(
    (densities - densities.min()) / (densities.max() - densities.min())
)
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh], mesh_show_back_face=True)
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)
# print(mesh)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
"""

# add noise
# print("create noisy mesh")
# vertices = np.asarray(mesh.vertices)
# noise = 0.0001
# vertices += np.random.uniform(0, noise, size=vertices.shape)
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# filtering
print("filter with average with 1 iteration")
mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([pcl, mesh_out], mesh_show_back_face=True)
