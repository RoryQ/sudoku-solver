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

def cross(A, B):
    """Cross product of elements in A and elements in B"""
    return [a + b for a in A for b in B]

def square_coordinates(bottom_left, width):
    (x, y), w = bottom_left, width
    return [x, y], [x, y+w], [x+w, y+w], [x+w, y]

"""
Grid elements will be referenced as follows:

A1 A2 A3| A4 A5 A6| A7 A8 A9
B1 B2 B3| B4 B5 B6| B7 B8 B9
C1 C2 C3| C4 C5 C6| C7 C8 C9
--------+---------+---------
D1 D2 D3| D4 D5 D6| D7 D8 D9
E1 E2 E3| E4 E5 E6| E7 E8 E9
F1 F2 F3| F4 F5 F6| F7 F8 F9
--------+---------+---------
G1 G2 G3| G4 G5 G6| G7 G8 G9
H1 H2 H3| H4 H5 H6| H7 H8 H9
I1 I2 I3| I4 I5 I6| I7 I8 I9
"""
# Generate 4 co-ordinates for each element
bl_x = [a*(side/9.0) for a in range(9)]
bl_y = bl_x[::-1]
bl_grid = [(x, y) for y in bl_y for x in bl_x]
width = bl_x[1]
grid_coordinates = [np.array(square_coordinates(e, width), np.float32) for e in bl_grid]

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
grid_refs = cross(rows, cols)
grid_elements = dict(zip(grid_refs, grid_coordinates))

# write solution to image