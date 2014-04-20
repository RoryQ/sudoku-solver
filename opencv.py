import cv2
import numpy as np
from contours import Contour

img = cv2.imread('sudoku.jpg')

# remove background
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(imgray, 255, 1, 1, 11, 2)

# detect puzzle area
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
contours = [Contour(ctn) for ctn in contours]
contours.sort(key=lambda x: x.area, reverse=True)
biggest = contours[0]

# warp puzzle area into square
side = np.ceil(biggest.perimeter / 4).astype("i")
square = np.array([[0, 0], [0, side], [side, side], [side, 0]], np.float32)  # BL TL TR BR
trans = cv2.getPerspectiveTransform(biggest.approx.astype("f"), square)
warp = cv2.warpPerspective(imgray, trans, (side, side))

cv2.imwrite('warp.jpg', warp)
