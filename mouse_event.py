import sys
import cv2
import numpy as np
import math
from random import shuffle
import selectinwindow

# Set recursion limit
sys.setrecursionlimit(10 ** 9)

drawing = False
xi, yi = -1, -1
B = [i for i in range(256)]
G = [i for i in range(256)]
R = [i for i in range(256)]


def nothing(x):
    pass


# cv2.createTrackbar("HUE_MIN", "image", 0, 255, nothing)
# cv2.createTrackbar("HUE_MAX", "image", 0, 255, nothing)


def onMouse(event, x, y, flags, frame):
    global xi, yi, drawing, B, G, R
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        xi, yi = x, y
        shuffle(B), shuffle(G), shuffle(R)

    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing:
    #         cv2.rectangle(frame, (xi, yi), (x, y), (B[0], G[0], R[0]), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (xi, yi), (x, y), (B[0], G[0], R[0]), 3)


# frame = np.zeros((512, 512, 3), np.uint8)
rgb_image = np.load("rgb_image_raw.npy")
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", onMouse, param=rgb_image)

while True:
    cv2.imshow("frame", rgb_image)
    key = cv2.waitKey(1)
    if key == 27:
        break
    # h_min = cv2.getTrackbarPos("HUE_MIN", "image")
    # h_max = cv2.getTrackbarPos("HUE_MAX", "image")

cv2.destroyAllWindows()
