import random
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def projection_line(pcl, polygon_number=10):
    """
    get projection line
    input: point cloud
    output: vector, point
    """
    rad = 0.0125

    # get random point and normal vector
    point_cloud = np.asarray(pcl.points)
    normal_vectors = np.asarray(pcl.normals)
    num = point_cloud.shape[0]  # number of pcl
    rand = random.randrange(1, num + 1)
    p = point_cloud[rand].reshape(1, 3).T  # point
    n_vector = normal_vectors[rand].reshape(1, 3).T  # normal vector

    # a = [i, 0, 0]
    a = np.zeros((3, 1))  # point
    i = (n_vector[1] * p[1] + n_vector[2] * p[2]) / n_vector[0] + p[0]
    a[:1, :] = i

    # get normalized vector
    x_vector = a - p
    x_vector = x_vector / np.linalg.norm(x_vector)
    y_vector = np.cross(n_vector.T, x_vector.T).T
    y_vector = y_vector / np.linalg.norm(y_vector)

    polygon_angle = np.linspace(0.0, 360.0, num=polygon_number, endpoint=False)
    polygon_points = []

    for theta in polygon_angle:
        polygon_vector = rad * (x_vector * np.cos(theta) + y_vector * np.sin(theta))
        polygon_points.append(p + polygon_vector)

    return polygon_points


points = np.load("pcl_test/pcl1.npy")
print(points.shape)
pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(points)

# estimate normal
pcl.normals = o3d.utility.Vector3dVector(
    np.zeros((1, 3))
)  # invalidate existing normals
pcl.estimate_normals()
# o3d.visualization.draw_geometries([pcl], point_show_normal=True)
point = np.asarray(pcl.points)
print("pcl:", point[100].reshape(1, 3).T)
print("num:", np.asarray(pcl.normals).shape, np.asarray(pcl.points).shape)

distances = pcl.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist
radius = 1.2 * avg_dist
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcl, o3d.utility.DoubleVector([radius, radius * 2])
)
print(mesh)
# o3d.visualization.draw_geometries([pcl, mesh])
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

print(np.asarray(mesh.vertices))
print(np.asarray(mesh.triangles))
print(np.asarray(pcl.normals))


print("filter with average with 1 iteration")
mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([pcl, mesh_out], mesh_show_back_face=True)

# test
print("test\n", projection_line(pcl))
