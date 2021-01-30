import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

depth_inpaint = np.load("depth_image_inpaint.npy")
color_raw = np.load("rgb_image_raw.npy")
depth_raw = np.load("depth_image_raw.npy")

rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)
# print(rgbd_image)

plt.subplot(1, 2, 1)
plt.title("TUM grayscale image")
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title("TUM depth image")
plt.imshow(rgbd_image.depth)
plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    ),
)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd], zoom=0.35)

