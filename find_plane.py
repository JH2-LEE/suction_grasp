import cv2
import numpy as np
import struct

from matplotlib import pyplot as plt

import pyransac3d as pyrsc
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D


class Segmentation:
    def __init__(self, rgb, depth):
        self.rgb_image = np.load(rgb)
        self.depth_image = np.load(depth)
        # plt.imshow(self.rgb_image)
        # plt.show()

    def rgb_segmentation(self):
        # points
        x = 460
        y = 0
        w = 410
        h = 660

        # image
        rgb_hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)  # H*w*3
        rgb_tmp = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)  # H*w*1

        # rgb_tf = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("rgb", rgb_tf)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        self.zero_mask = np.zeros_like(rgb_tmp, np.uint8)
        pts = np.array([[x, y], [x, y + h], [x + w, y + h], [x + w, y]], np.int32)
        roi_mask = cv2.fillPoly(self.zero_mask.copy(), [pts], (255))

        # chroma key
        min_green = (60 - 15, 30, 30)
        max_green = (60 + 15, 255, 255)
        img_mask = cv2.inRange(rgb_hsv, min_green, max_green)
        img_mask = np.invert(img_mask)

        mask = cv2.bitwise_and(roi_mask, img_mask)

        kernel = np.ones((5, 5), np.uint8)
        closing_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return closing_mask

    def split_objects(self, img):
        roi_mask1 = cv2.rectangle(
            self.zero_mask.copy(), (535, 200), (647, 320), 255, -1
        )
        roi_mask2 = cv2.rectangle(
            self.zero_mask.copy(), (662, 241), (783, 386), 255, -1
        )
        roi_mask3 = cv2.rectangle(
            self.zero_mask.copy(), (560, 325), (678, 475), 255, -1
        )

        roi_mask1 = cv2.bitwise_and(roi_mask1, img)
        roi_mask2 = cv2.bitwise_and(roi_mask2, img)
        roi_mask3 = cv2.bitwise_and(roi_mask3, img)

        return roi_mask1, roi_mask2, roi_mask3


class PointCloud:
    def __init__(self):
        self.camera_intrinsic = np.array(
            [[610.335, 0.0, 641.3], [0.0, 610.141, 610.141], [0.0, 0.0, 1.0]],
        )

    def deprojection(self, image):
        height, width = image.shape[0:2]
        pixel_point = np.zeros((height * width, 3), np.float32)
        depth = image.reshape(-1, 1)
        pixel_point[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * depth
        pixel_point[:, 2:] = depth

        point_cloud = np.dot(np.linalg.inv(self.camera_intrinsic), pixel_point.T)
        return point_cloud

    def delete_zero(self, pcl):
        pcl_temp = pcl[:, 2:]
        pcl = np.delete(pcl, np.where(pcl_temp == 0)[0], axis=0)
        return pcl


# seg = Segmentation("rgb_image_raw.npy", "depth_image_inpaint.npy")
seg = Segmentation("rgb_image_raw.npy", "depth_image_inpaint.npy")

rgb_mask = seg.rgb_segmentation()

object_mask1 = seg.split_objects(rgb_mask)[0]
object_mask2 = seg.split_objects(rgb_mask)[1]
object_mask3 = seg.split_objects(rgb_mask)[2]

cv2.imwrite("mask1.jpg", object_mask1)
cv2.imwrite("mask2.jpg", object_mask2)
cv2.imwrite("mask3.jpg", object_mask3)

# object1 = np.multiply(object_mask1, seg.depth_image.copy())
object1 = object_mask1 * seg.depth_image.copy() / 255
object2 = object_mask2 * seg.depth_image.copy() / 255
object3 = object_mask3 * seg.depth_image.copy() / 255


pc = PointCloud()
point_cloud1 = pc.deprojection(object1).T
point_cloud2 = pc.deprojection(object2).T
point_cloud3 = pc.deprojection(object3).T
point_cloud = pc.deprojection(seg.depth_image).T

np.save("pc1.npy", point_cloud1)
np.save("pc2.npy", point_cloud2)
np.save("pc3.npy", point_cloud3)
np.save("pc.npy", point_cloud)

pcl = pc.delete_zero(point_cloud)
pcl1 = pc.delete_zero(point_cloud1)
pcl2 = pc.delete_zero(point_cloud2)
pcl3 = pc.delete_zero(point_cloud3)


plane1 = pyrsc.Plane()
plane2 = pyrsc.Plane()
plane3 = pyrsc.Plane()
best_eq1, best_inliers1 = plane1.fit(pcl1, 0.005)
best_eq2, best_inliers2 = plane1.fit(pcl2, 0.005)
best_eq3, best_inliers3 = plane1.fit(pcl3, 0.005)


pcl1_inlier = pcl1[best_inliers1]
pcl2_inlier = pcl2[best_inliers2]
pcl3_inlier = pcl3[best_inliers3]

# add rgb colors in pcl1

np.save("pcl1.npy", pcl1)
np.save("pcl2.npy", pcl2)
np.save("pcl3.npy", pcl3)


# x = np.linspace(-1, 1, 10)
# y = np.linspace(-1, 1, 10)
# X, Y = np.meshgrid(x, y)
# coeff = [
#     0.003137052191244864,
#     0.11486620490005427,
#     0.9933760183713961,
#     -0.6998064259710848,
# ]
# Z = (
#     (-0.003137052191244864 / 0.9933760183713961) * X
#     + (-0.11486620490005427 / 0.9933760183713961) * Y
#     + (0.6998064259710848 / 0.9933760183713961)
# )
# fig = plt.figure()
# ax = fig.gca(projection="3d")
# surf = ax.plot_surface(X, Y, Z)
# plt.show()

