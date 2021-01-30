import numpy as np
import pyransac3d as pyrsc
import open3d as o3d


def cube(x=1, dx=2):
    pcl = np.empty((0, 3))

    # xx = np.linspace(x, x + dx, 40)
    # yy = np.linspace(y, y + dy, 40)
    # zz = np.linspace(z, z + dz, 40)

    # X, Y = np.meshgrid(xx, yy)how to make cylinder in point cloud

    mesh = np.mgrid[x : x + dx : 40j, x : x + dx : 40j].T.reshape(-1, 2)
    surf1 = np.hstack((mesh, np.full((mesh.shape[0], 1), x)))
    surf2 = np.hstack((mesh, np.full((mesh.shape[0], 1), x + dx)))
    surf = np.vstack((surf1, surf2))

    pcl = np.r_[pcl, surf]
    pcl = np.r_[pcl, np.transpose([surf[..., 2], surf[..., 1], surf[..., 0]])]
    pcl = np.r_[pcl, np.transpose([surf[..., 1], surf[..., 2], surf[..., 0]])]

    return pcl


def cylinder(x=1, h=3):
    pcl = np.empty((0, 3))
    r = np.linspace(0, x, 20)
    t = np.linspace(0, 2 * np.pi, 60)

    # base
    mesh_polar = np.mgrid[0:x:20j, 0 : 2 * np.pi : 60j].T.reshape(-1, 2)
    mesh_cartesian = np.hstack(
        (
            mesh_polar[:, :1] * np.cos(mesh_polar[:, 1:]),
            mesh_polar[:, :1] * np.sin(mesh_polar[:, 1:]),
        )
    )
    surf1 = np.hstack((mesh_cartesian, np.full((mesh_cartesian.shape[0], 1), 0)))
    surf2 = np.hstack((mesh_cartesian, np.full((mesh_cartesian.shape[0], 1), 3)))
    pcl = np.r_[pcl, surf1]
    pcl = np.r_[pcl, surf2]

    # side
    r = np.linspace(1, 2, 1)
    t = np.linspace(0, 2 * np.pi, 60)
    z = np.linspace(0, h, 50)
    [R, T, Z] = np.meshgrid(r, t, z)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    surf = np.transpose([X, Y, Z]).reshape(-1, 3)
    pcl = np.r_[pcl, surf]
    return pcl


"""
def add_noise(pcl):
    sigma = 0.1
    noise = np.sqrt(sigma) * np.random.normal(3, 6.25, size(pcl))
    pcl = pcl 
"""


def delete_zero(pcl):
    pcl_temp = pcl[:, 2:]
    pcl = np.delete(pcl, np.where(pcl_temp == 0)[0], axis=0)
    return pcl


pcl_cube = cube()
pcl_cylinder = cylinder()

np.save("pcl_cube.npy", pcl_cube)
np.save("pcl_cylinder.npy", pcl_cylinder)

plane_cube = pyrsc.Plane()
plane_cylinder = pyrsc.Plane()

best_eq_cube, best_inliers_cube = plane_cube.fit(pcl_cube, 0.005)
best_eq_cylinder, best_inliers_cylinder = plane_cylinder.fit(pcl_cylinder, 0.005)

pcl_cube_inlier = pcl_cube[best_inliers_cube]
pcl_cylinder_inlier = pcl_cylinder[best_inliers_cylinder]

np.save("pcl_cube_inlier.npy", pcl_cube_inlier)
np.save("pcl_cylinder_inlier.npy", pcl_cylinder_inlier)

