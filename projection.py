import random
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import time


def find_moment_pixel(img):
    M = cv2.moments(img, True)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


class SuctionGrasp:
    def __init__(self, center, info):
        center_width, center_height = center
        pcl_index = info[0]
        self.polygon_points = info[1]
        self.normal_vector = info[2]

        width = pcl_index % 1920
        height = pcl_index // 1920
        self.distance = (width - center_width) * (width - center_width) + (
            height - center_height
        ) * (height - center_height)

    @property
    def apex(self):
        ret = self.target + self.normal_vector
        return ret


apex = SuctionGrasp.apex


class LocalGeometry:
    def __init__(self, pcl, inner_pcl, mesh):
        self.point_cloud = np.asarray(pcl.points)
        self.normal_vectors = np.asarray(pcl.normals)
        self.inner_point_cloud = inner_pcl
        self.mesh_vertices = np.asarray(mesh.vertices)
        self.mesh_triangles = np.asarray(mesh.triangles)
        # print("self.point_cloud\n", self.point_cloud)
        # print("self.mesh_vertices\n", self.mesh_vertices)
        pass

    def projection_line(self, polygon_number):
        """
        get projection line
        input: point cloud
        output: vector, point
        """
        rad = 0.0125
        self.polygon_number = polygon_number

        self.real_perimeter_distance = 2 * rad * np.sin(np.pi / polygon_number)
        self.h = rad
        self.real_cone_distance = np.sqrt(2) * rad
        # get random point and normal vector
        # num = self.point_cloud.shape[0]  # number of pcl
        # self.rand = random.randrange(1, num + 1)
        inner_idx_arr = np.where(self.inner_point_cloud[:, 2:] != 0)[0]
        num = inner_idx_arr.shape[0]
        rand_idx = random.randrange(1, num + 1)
        self.rand = inner_idx_arr[rand_idx]  # index of pcl

        p = self.point_cloud[self.rand].reshape(1, 3).T  # point
        self.target_point = p
        n_vector = self.normal_vectors[self.rand].reshape(1, 3).T  # normal vector

        # change normal to get apex
        norm = np.sqrt((n_vector * n_vector).sum())
        n_vector = n_vector / norm
        camera_vector = np.array([[0, 0, 1]])
        if (camera_vector * n_vector).sum() > 0:
            n_vector = -n_vector

        self.line_normal = n_vector

        # a = [i, 0, 0]
        a = np.zeros((3, 1))  # point
        i = (n_vector[1] * p[1] + n_vector[2] * p[2]) / n_vector[0] + p[0]
        a[:1, :] = i

        # get normalized vector
        x_vector = a - p
        x_vector = x_vector / np.linalg.norm(x_vector)
        y_vector = np.cross(n_vector.T, x_vector.T).T
        y_vector = y_vector / np.linalg.norm(y_vector)

        polygon_angle = np.linspace(0.0, 2 * np.pi, num=polygon_number, endpoint=False)
        self.polygon_points = []
        for theta in polygon_angle:
            polygon_vector = rad * (x_vector * np.cos(theta) + y_vector * np.sin(theta))
            self.polygon_points.append(p + polygon_vector)
        # print("polygon_points\n", polygon_points)
        return self.polygon_points

    def image_mask(self, rad=20, height=1080, width=1920):
        self.raw_index = np.load("raw_index/raw_index1.npy")
        pcl_index = self.raw_index[self.rand]
        self.pcl_index = pcl_index
        h = pcl_index // width
        w = pcl_index % width
        binary_mask = np.zeros((height, width))
        # draw circle in (x,y)
        binary_mask = cv2.circle(binary_mask, (w, h), rad, (255), thickness=-1) / 255
        # zero delete index
        local_idx_tmp = np.where(binary_mask.reshape(-1, 1) != 0)[0]
        local_idx_tmp = local_idx_tmp.reshape((local_idx_tmp.shape[0], 1))

        self.local_idx = np.empty((0, 1), dtype=int)
        for i in local_idx_tmp:
            index = np.where(self.raw_index == i)[0]
            # print(self.local_idx)
            try:
                self.local_idx = np.vstack((self.local_idx, index))
            except:
                pass
        # print(local_idx)
        return self.local_idx
        # pass

    def mesh_to_plane(self):
        local_idx_copy = self.local_idx
        log_arr = np.isin(self.mesh_triangles, local_idx_copy.flatten())
        # print(log_arr.shape[1])
        # print(log_arr.sum(axis=0))
        log_arr_sum = np.ones((log_arr.shape[0], 1), dtype=bool)
        for i in range(log_arr.shape[1]):  # column vector
            col_vec = log_arr[:, i : i + 1]
            log_arr_sum = np.logical_and(log_arr_sum, col_vec)
        # print(np.where(log_arr_sum == True)[0].shape[0])
        idx_true = np.where(log_arr_sum == True)[0]
        self.local_triangle = self.mesh_triangles[idx_true, :]
        # print(self.local_triangle)
        self.mesh_point1 = self.mesh_vertices[self.local_triangle[:, 0]]
        self.mesh_point2 = self.mesh_vertices[self.local_triangle[:, 1]]
        self.mesh_point3 = self.mesh_vertices[self.local_triangle[:, 2]]

        # return self.local_triangle
        pass

    def find_intersection_point(self):
        mesh_vec1 = self.mesh_point1 - self.mesh_point2
        mesh_vec2 = self.mesh_point1 - self.mesh_point3
        self.mesh_normal = np.cross(mesh_vec1, mesh_vec2)
        # self.mesh_normal = (
        #     mesh_normal
        #     / np.sqrt((mesh_normal * mesh_normal).sum(axis=1))
        #     .reshape(1, mesh_normal.shape[0])
        #     .T
        # )

        self.intersection_points = []
        for polygon_point in self.polygon_points:
            polygon_point_arr = np.tile(polygon_point.T, (self.mesh_normal.shape[0], 1))
            line_normal_arr = np.tile(
                self.line_normal.T, (self.mesh_normal.shape[0], 1)
            )
            # print(polygon_point_arr)
            # print(line_normal_arr)
            param = np.multiply(
                self.mesh_normal, (self.mesh_point1 - polygon_point_arr)
            ).sum(axis=1) / np.multiply(self.mesh_normal, line_normal_arr).sum(axis=1)
            param = param.reshape(1, param.shape[0]).T  # t
            # print("param\n", param)
            intersection_point = (
                np.tile(param, (1, 3)) * line_normal_arr + polygon_point_arr
            )  # (x, y, z)
            self.intersection_points.append(intersection_point)
            # print(intersection_point)
        # print(self.intersection_points)
        # pass
        return self.intersection_points

    def plane_eq(self, normal_vec, plane_point, xyz):
        # for inner triangle func
        plane = np.multiply(normal_vec, (xyz - plane_point)).sum(axis=1)
        # print(plane.shape)
        plane = plane.reshape(1, plane.shape[0]).T
        return plane

    def sign(self, arr):
        arr_copy = arr.copy()
        arr_copy[arr_copy > 0] = 1
        arr_copy[arr_copy < 0] = -1
        return arr_copy

    def is_inner_triangle(self):
        # to check sign
        center_point = (self.mesh_point1 + self.mesh_point2 + self.mesh_point3) / 3

        line_vector = np.tile(self.line_normal.T, (self.mesh_point1.shape[0], 1))
        mesh_vector1 = self.mesh_point1 - self.mesh_point2
        mesh_vector2 = self.mesh_point2 - self.mesh_point3
        mesh_vector3 = self.mesh_point3 - self.mesh_point1

        # print(self.mesh_point1)

        mesh_plane_normal1 = np.cross(line_vector, mesh_vector1)
        mesh_plane_normal2 = np.cross(line_vector, mesh_vector2)
        mesh_plane_normal3 = np.cross(line_vector, mesh_vector3)

        self.mesh_plane_normal1 = (
            mesh_plane_normal1
            / np.sqrt((mesh_plane_normal1 * mesh_plane_normal1).sum(axis=1))
            .reshape(1, mesh_plane_normal1.shape[0])
            .T
        )
        self.mesh_plane_normal2 = (
            mesh_plane_normal2
            / np.sqrt((mesh_plane_normal2 * mesh_plane_normal2).sum(axis=1))
            .reshape(1, mesh_plane_normal2.shape[0])
            .T
        )
        self.mesh_plane_normal3 = (
            mesh_plane_normal3
            / np.sqrt((mesh_plane_normal3 * mesh_plane_normal3).sum(axis=1))
            .reshape(1, mesh_plane_normal3.shape[0])
            .T
        )

        inner_triangle = np.hstack(
            (
                self.plane_eq(self.mesh_plane_normal1, self.mesh_point1, center_point),
                self.plane_eq(self.mesh_plane_normal2, self.mesh_point2, center_point),
                self.plane_eq(self.mesh_plane_normal3, self.mesh_point3, center_point),
            )
        )
        inner_triangle_sign = self.sign(inner_triangle)

        self.inner_points = np.empty((0, 3))
        for point in self.intersection_points:
            iter_point = np.hstack(
                (
                    self.plane_eq(
                        self.mesh_plane_normal1,
                        self.mesh_point1,
                        point,
                    ),
                    self.plane_eq(
                        self.mesh_plane_normal2,
                        self.mesh_point2,
                        point,
                    ),
                    self.plane_eq(
                        self.mesh_plane_normal3,
                        self.mesh_point3,
                        point,
                    ),
                )
            )
            iter_point_sign = self.sign(iter_point)
            # print("iter point\n", iter_point)
            is_inner = inner_triangle_sign + iter_point_sign
            is_inner[is_inner == 0] = False
            is_inner[is_inner != 0] = True
            is_inner = is_inner.sum(axis=1)
            # print(np.where(is_inner == 3)[0].shape[0])
            idx = np.where(is_inner == 3)[0]
            # print(iter_point[idx, :])  # deepcopy
            self.inner_points = np.vstack((self.inner_points, point[idx, :]))
            # print(iter_point)
        # print(inner_points)
        # print(self.inner_points.shape)
        # print("Inner Triangel\n", inner_triangle)
        # print("Iterable Point\n", iter_point)

        return self.inner_points

    def cal_distance(self):
        point_num = self.inner_points.shape[0]
        if point_num == self.polygon_number:
            inner_points = self.inner_points.copy()

            # perimeter
            inner_points_next = np.vstack(
                (inner_points[point_num - 1 :, :], inner_points[: point_num - 1, :])
            )
            inner_points_err = inner_points - inner_points_next
            # print(inner_points_err)
            inner_points_distance = np.sqrt(
                (inner_points_err * inner_points_err).sum(axis=1)
            )

            # cone
            target_point_array = np.tile(self.target_point.T, (point_num, 1))
            normal_vector_array = np.tile(self.line_normal.T, (point_num, 1))
            t_arr = ((inner_points - target_point_array) * normal_vector_array).sum(
                axis=1
            )
            t_val = min(t_arr.sum() / point_num - self.h, 0)
            # print("t_val:", t_val)
            self.apex_point = self.target_point - t_val * self.line_normal
            # print("apex:", self.apex_point.T)
            apex_array = np.tile(self.apex_point.T, (point_num, 1))
            apex_points_err = inner_points - apex_array
            cone_points_distance = np.sqrt(
                (apex_points_err * apex_points_err).sum(axis=1)
            )

            count = 0
            for distance in inner_points_distance:
                strain = (
                    (distance - self.real_perimeter_distance)
                    * 100
                    / self.real_perimeter_distance
                )
                if strain >= 10:
                    count += 1
            for distance in cone_points_distance:
                strain = (
                    (distance - self.real_cone_distance) * 100 / self.real_cone_distance
                )
                if strain >= 10:
                    count += 1

            if count == 0:
                # print("ok")
                return self.pcl_index, self.inner_points, self.line_normal.T
            else:
                pass
        else:
            pass


points = np.load("pcl/pcl1.npy")
inner_points = np.load("inner/inner_pcl1.npy")
# print(points.shape)
# print(inner_points.shape)
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
rgb_mask = cv2.imread("pcl/rgb_mask1.png")
cx, cy = find_moment_pixel(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY))

# start
start = time.time()
lg = LocalGeometry(pcl, inner_points, mesh)
polygon_number = 36
grasps = []
centroid = []
for i in range(20):
    lg.projection_line(polygon_number)
    lg.image_mask()
    lg.mesh_to_plane()
    lg.find_intersection_point()
    lg.is_inner_triangle()
    # polygon_pcl = lg.is_inner_triangle()
    # polygon_pcl = o3d.geometry.PointCloud()
    # polygon_pcl.points = o3d.utility.Vector3dVector(lg.is_inner_triangle())
    # o3d.visualization.draw_geometries([polygon_pcl, mesh_out], mesh_show_back_face=True)
    ret = lg.cal_distance()
    if ret is not None:
        # print(ret)
        # print("loading")
        grasp = SuctionGrasp((cx, cy), ret)
        grasps.append([grasp.polygon_points, grasp.normal_vector])
        centroid.append(grasp.distance[0])
centroid = np.asarray(centroid)
print(grasps)
print(centroid)
# print(centroid.shape)
print(centroid[np.argmin(centroid)])
print(grasps[np.argmin(centroid)])
# end
end = time.time()
# print("time:", end - start)
# f = open("polygon_18.txt", "a")
# f.write("{}\n".format(end - start))
# f.close()

polygon_pcl = o3d.geometry.PointCloud()
polygon_pcl.points = o3d.utility.Vector3dVector(grasps[np.argmin(centroid)][0])
apex = o3d.geometry.PointCloud()
apex.points = o3d.utility.Vector3dVector(lg.apex_point.T)
o3d.visualization.draw_geometries(
    [polygon_pcl, apex, mesh_out], mesh_show_back_face=True
)
