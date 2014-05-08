import cv2
import numpy as np
from contours import Contour
from PIL import Image
from PIL.ExifTags import TAGS
from sys import exit


def get_exif_data(fname):
    ret = {}
    try:
        img = Image.open(fname)
        if hasattr(img, '_getexif'):
            exifinfo = img._getexif()
            if exifinfo != None:
                for tag, value in exifinfo.items():
                    decoded = TAGS.get(tag, tag)
                    ret[decoded] = value
    except IOError:
        print 'IOERROR' + fname
    return ret


def square_coordinates(bottom_left, width):
    (x, y), w = bottom_left, width
    return [x, y], [x, y+w], [x+w, y+w], [x+w, y]  # BL TL TR BR


def cross(A, B):
    """Cross product of elements in A and elements in B"""
    return [a + b for a in A for b in B]
fn = 'sudoku3.jpg'
exif = get_exif_data(fn)
rot = 0
if "Orientation" in exif:
    orientation = exif["Orientation"]
    if orientation == 8:
        rot = 1
    elif orientation == 3:
        rot = 2
    elif orientation == 6:
        rot = 3

img = cv2.imread(fn)
if rot > 0:
    img = np.rot90(img.copy(), rot)
img = img.copy()

pixelk = 15
# remove background
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(imgray, (pixelk, pixelk), 0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
closed = cv2.morphologyEx(gauss, cv2.MORPH_CLOSE, kernel)
divide = cv2.divide(gauss.astype("f"), closed.astype("f"))
norm = divide.copy()
cv2.normalize(divide, norm, 0, 255, cv2.NORM_MINMAX)
_, thresh = cv2.threshold(norm.astype("uint8"), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [Contour(ctn) for ctn in contours]
contours.sort(key=lambda x: x.area, reverse=True)
for i in range(len(contours)):
    if len(contours[i].approx) == 4:
        biggest = contours[i]
        break

#for cnt in contours:
#    cnt.draw_stuff(img)
biggest.draw_stuff(img)
cv2.imwrite('biggest.jpg', img)

mask = thresh.copy()
mask.fill(0)
cv2.drawContours(mask, [biggest.cnt], 0, 255, -1)
cv2.drawContours(mask, [biggest.cnt], 0, 0, 2)
bit = cv2.bitwise_and(thresh, mask)

kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
dx = cv2.Sobel(thresh, cv2.CV_16S, 1, 0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)



# warp puzzle area into square
side = np.ceil(biggest.perimeter / 4).astype("i")
square = np.array(square_coordinates((0, 0), side), np.float32)
trans = cv2.getPerspectiveTransform(biggest.bbox.astype("f"), square)
warp = cv2.warpPerspective(thresh, trans, (side, side))

cv2.imwrite('warp.jpg', warp)



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

Contours have a bottom left origin, whereas images are top left.
I1 has a bottom left contour origin of (0,0)
"""
# Generate 4 co-ordinates for each element
bl = [a*(side/9.0) for a in range(9)]
bl_grid = [(x, y) for y in bl for x in bl]
width = bl[1]
grid_coordinates = [np.array(square_coordinates(e, width), np.float32) for e in bl_grid]

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
grid_refs = cross(rows, cols)
grid_elements = dict(zip(grid_refs, grid_coordinates))


side = width.astype("i")
square = np.array(square_coordinates((0, 0), side), np.float32)

for i in grid_elements:
    cnt = Contour(grid_elements[i])
    halfbox = cnt.shrinkbox(12).astype("f")
    trans = cv2.getPerspectiveTransform(halfbox, square)
    box = cv2.warpPerspective(warp, trans, (side, side))
    blur = cv2.GaussianBlur(box, (5, 5), 0)
    (_, thresh) = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # contours = [Contour(ctn) for ctn in contours]
    # box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
    # for cnt in contours:
    #     cnt.draw_stuff(box)

    cv2.imwrite(i + '.tiff', box)

    if 1==0 and len(contours) > 0:
        print i
        contours.sort(key=lambda x: x.area, reverse=True)
        biggest = contours[0]
        side = np.ceil(biggest.perimeter / 4).astype("i")
        square = np.array(square_coordinates((0, 0), side), np.float32)
        _, _, width, height = biggest.bounding_box
        cnt = biggest.bbox.astype("f")
        trans = cv2.getPerspectiveTransform(cnt, cnt)
        warp = cv2.warpPerspective(box, trans, (width, height))
        cv2.imwrite(i + '.tiff', warp)

    # write solution to image