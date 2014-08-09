import cv2
import numpy as np
from contours import Contour
from PIL import Image
from PIL.ExifTags import TAGS
import argparse
import pymorph

DEBUG = False

def get_exif_data(fname):
    """read image from file name and retrieve exif data"""
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


def get_image(filename):
    exif = get_exif_data(filename)
    rot = 0
    if "Orientation" in exif:
        orientation = exif["Orientation"]
        if orientation == 8:
            rot = 1
        elif orientation == 3:
            rot = 2
        elif orientation == 6:
            rot = 3
    img = cv2.imread(filename)
    if rot > 0:
        img = np.rot90(img.copy(), rot)
    return downsize_image(img)

def downsize_image(img):
    if sum(img.shape[:2]) > 1500:
        small = map((lambda a: int(a /(sum(img.shape[:2]) / 1500.0))), 
                    img.shape[:2])
        return cv2.resize(img, tuple(small))
    return img

def remove_image_shading(image, pixelk=None, ksize=None):
    # apply morphological closing on image to extract background
    # then divide original image to remove background shading
    if pixelk is None: 
        pixelk = 3
    if ksize is None: 
        ksize = 17
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(imgray, (pixelk, pixelk), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(gauss, cv2.MORPH_CLOSE, kernel)
    divide = cv2.divide(gauss.astype("f"), closed.astype("f"))
    norm = divide.copy()
    cv2.normalize(divide, norm, 0, 255, cv2.NORM_MINMAX)
    return norm


def get_contours(imag):
    contours, _ = cv2.findContours(imag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [Contour(ctn) for ctn in contours]
    contours.sort(key=lambda x: x.area, reverse=True)
    return contours

def get_biggest_rect(contours):
    for i in range(len(contours)):
        if len(contours[i].approx) == 4:
            return contours[i]

def sobel_filter(image, xdirection=True, ksize=None):
    if xdirection is True:
        x, y = 1, 0
    else:
        x, y = 0, 1
    if ksize is None: 
        ksize = 7
    dx = cv2.Sobel(image, cv2.CV_16S, x, y, ksize=ksize)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    return dx

def warp_to_contour(image, contour):
    side = contour.side
    square = np.array(square_coordinates((0, 0), side), np.float32)
    trans = cv2.getPerspectiveTransform(contour.bbox.astype("f"), square)
    return cv2.warpPerspective(image, trans, (side, side))

def square_contour_to_coords(contour):
    bl = [a*(contour.side/9.0) for a in range(9)]
    bl_grid = [(x, y) for y in bl for x in bl]
    width = bl[1]
    side = width.astype("i")
    square = np.array(square_coordinates((0, 0), side), np.float32)
    grid_elements = [np.array(square_coordinates(e, width), np.float32) for e in bl_grid]
    return grid_elements, square, side
    
def generate_grid_elements(contour):
    """Generate 4 co-ordinates for each element """
    grid_coordinates, _, _ = square_contour_to_coords(contour)
    digits = '123456789'
    rows = 'ABCDEFGHI'
    cols = digits
    grid_refs = cross(rows, cols)
    return dict(zip(grid_refs, grid_coordinates))

def get_args():
    """get args or display args if missing"""
    parser = argparse.ArgumentParser(description="OpenCV Sudoku Processing")
    parser.add_argument('-f', '--file', help='Image file to process',
            required=True)
    parser.add_argument('-d', '--debug', help='Output debug images',
            action='store_true')
    return parser.parse_args()

def thresh_img(img):
    img_noshading = remove_image_shading(img)
    if DEBUG:
        cv2.imwrite('no_shading.jpg', img_noshading)
    _, thresh = cv2.threshold(img_noshading.astype("uint8"), 128, 255, 
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

def main():
    args = get_args()
    global DEBUG
    DEBUG = args.debug
    fn = args.file.split(".")[0]
    img = get_image(args.file)
    thresh = thresh_img(img)
    contours = get_contours(thresh.copy())
    biggest = get_biggest_rect(contours)

    warp = warp_to_contour(thresh, biggest)
    grid_coordinates, square, side = square_contour_to_coords(biggest)
    grid_elements = generate_grid_elements(biggest)

    if DEBUG:
        allimg = img.copy()
        for c in contours:
            c.draw_stuff(allimg)
        biggest.draw_stuff(img)
        cv2.imwrite('all.jpg', allimg)
        cv2.imwrite('biggest.jpg', img)
        cv2.imwrite('thresh.jpg', thresh)
        cv2.imwrite('warp.jpg', warp)

    for elem in grid_elements:
        cnt = Contour(grid_elements[elem])
        halfbox = cnt.shrinkbox(100).astype("f")
        trans = cv2.getPerspectiveTransform(halfbox, square)
        box = cv2.warpPerspective(warp, trans, (side, side))
        binary = pymorph.binary(box)
        box = binary.astype(int) * 255
        box = pymorph.edgeoff(box)
        #pdb.set_trace()
        if DEBUG:
            box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
            for cnt in contours:
                cnt.draw_stuff(box)

        cv2.imwrite(fn + " - " + elem + '.jpg', box)

if __name__ == '__main__':
    main()

