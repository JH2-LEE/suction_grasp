#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import cv2

if __name__ == "__main__":
    depth_image_raw = np.load("depth_image_raw.npy")
    rgb_image_raw = np.load("rgb_image_raw.npy")
    depth_image_inpaint = np.load("depth_image_inpaint.npy")
    raw_idx = np.load("raw_index/raw_index1.npy")
    print(raw_idx)
    # cv2.imshow("depth_image_raw", depth_image_raw)
    # cv2.imshow("rgb_image_raw", rgb_image_raw)
    # cv2.imshow("depth_image_inpaint", depth_image_inpaint)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.figure()
    plt.imshow(depth_image_inpaint)
    plt.figure()
    plt.imshow(depth_image_raw)
    plt.show()

cv2.cvtColor(rgb_image_raw, cv2.COLOR_BGR2RGB)