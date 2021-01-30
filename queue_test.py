import cv2
import numpy as np
from collections import deque

dq = deque([])

for i in range(1, 18):
    image = cv2.imread("pcl/rgb_mask{0}.png".format(i))
    if i == 1:
        dq.append(image)
        dq.append(image)
        dq.append(image)
    dq.append(image)
    dq.popleft()
    print(dq)
