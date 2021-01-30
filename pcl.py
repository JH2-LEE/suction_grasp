import numpy as np
from matplotlib import pyplot as plt

pc_raw = np.load("pc_raw.npy")
pc_inpaint = np.load("pc_inpaint.npy")
print(pc_raw)


def slicing(pcl):
    # height, width = pcl.shape[0:2]
    pcl = pcl.reshape(-1, 2, 3)[:, :1, :]

    pcl = pcl.reshape(720, 640, 3)
    pcl = pcl.transpose(1, 0, 2)
    pcl = pcl.reshape(-1, 2, 3)[:, :1, :]
    # .T.reshape(-1, 2)[:, :1].reshape(-1, 1)
    return pcl.reshape(-1, 3)


new_pc_raw = slicing(pc_raw)
new_pc_inpaint = slicing(pc_inpaint)

np.save("new_pc_raw.npy", new_pc_raw)
np.save("new_pc_inpaint.npy", new_pc_inpaint)

