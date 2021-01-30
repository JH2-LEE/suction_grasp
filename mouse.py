import numpy as np
import cv2
from random import shuffle
import math

mode, drawing = True, False
xi, yi = -1, -1
B = [i for i in range(256)]
G = [i for i in range(256)]
R = [i for i in range(256)]


def onMouse(event, x, y, flags, frame):
    global xi, yi, drawing, mode, B, G, R
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        xi, yi = x, y
        shuffle(B), shuffle(G), shuffle(R)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(frame, (xi, yi), (x, y), (B[0], G[0], R[0]), -1)
            else:
                r = (xi - x) ** 2 + (yi - y) ** 2
                r = int(math.sqrt(r))
                cv2.circle(frame, (xi, yi), r, (B[0], G[0], R[0]), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(frame, (xi, yi), (x, y), (B[0], G[0], R[0]), -1)
        else:
            r = (xi - x) ** 2 + (yi - y) ** 2
            r = int(math.sqrt(r))
            cv2.circle(frame, (xi, yi), r, (B[0], G[0], R[0]), -1)


frame = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", onMouse, param=frame)

while True:
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("m"):
        mode = not mode
cv2.destroyAllWindows()
