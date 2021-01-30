import numpy as np
import open3d as o3d

# import the point cloud using numpy and store as 03d object
point_cloud = np.load("pcl1.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
# Downsample with a voxel size of 10 cm
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
# estimate normals of points
voxel_down_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=300)
)
# plot the point cloud
o3d.visualization.draw_geometries([voxel_down_pcd])
radii = [0.25, 0.5]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    voxel_down_pcd, o3d.utility.DoubleVector(radii)
)
# plot the mesh
o3d.visualization.draw_geometries([rec_mesh])