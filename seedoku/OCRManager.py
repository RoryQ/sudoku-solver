import cv2
import numpy as np
from contours import Contour
from PIL import Image
from PIL.ExifTags import TAGS
import pymorph as pm
import mahotas as mh
import gzip
import cStringIO
import os
import cPickle
from skimage import morphology as mrp
from mytinyocr.feature.resize import Resize

class OCRManager(object):

    def __init__(self, ocr_alg, feature_alg=None):
        if feature_alg is None:
            feature_alg = Resize()
        self.feature_alg = feature_alg
        self.ocr_alg = ocr_alg

    def Image_to_puzzle(self, img):
        downsize = self._downsize_image(img)
        thresh = self._thresh_img(downsize)
        contours = self._get_contours(thresh.copy())
        biggest = self._get_biggest_rect(contours)

        warp = self._warp_to_contour(thresh, biggest)
        _, square, side = self._square_contour_to_coords(biggest)
        grid_elems = self._generate_grid_elements(biggest)
        puzzle = []
        for elem in grid_elems:
            cnt = Contour(grid_elems[elem])
            halfbox = cnt.shrinkbox(100).astype("f")
            trans = cv2.getPerspectiveTransform(halfbox, square)
            box = cv2.warpPerspective(warp, trans, (side, side))
            _, thresh = cv2.threshold(box, 127, 255, 0)
            binary = pm.binary(thresh)
            box = binary.astype(int) * 255
            box = pm.edgeoff(box)
            box = mh.croptobbox(box)

            if reduce(lambda x, y: x*y, box.shape) < 500:
                continue
            cv2.imwrite(elem + '.png', box)
            box = self._MNIST_preprocess(pm.binary(box).astype("float32"))
            features = self.feature_alg.get_features(box)
            res = self.ocr_alg.predict(features)
            puzzle.append(res)

        return puzzle

    def _square_coordinates(self, bottom_left, width):
        (x, y), w = bottom_left, width
        return [x, y], [x, y+w], [x+w, y+w], [x+w, y]  # bl tl tr br
    
    def _cross(self, a, b):
        """cross product of elements in a and elements in b"""
        return [a + b for a in a for b in b]
    
    def _downsize_image(self, img):
        if sum(img.shape[:2]) > 1500:
            small = map((lambda a: int(a /(sum(img.shape[:2]) / 1500.0))), 
                        img.shape[:2])
            return cv2.resize(img, tuple(small))
        return img
    
    def _remove_image_shading(self, image, pixelk=None, ksize=None):
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
    
    
    def _get_contours(self, imag):
        contours, _ = cv2.findContours(imag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [Contour(ctn) for ctn in contours]
        contours.sort(key=lambda x: x.area, reverse=True)
        return contours
    
    def _get_biggest_rect(self, contours):
        for i in range(len(contours)):
            if len(contours[i].approx) == 4:
                return contours[i]
    
    def _sobel_filter(self, image, xdirection=True, ksize=None):
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
    
    def _warp_to_contour(self, image, contour):
        side = contour.side
        square = np.array(self._square_coordinates((0, 0), side), np.float32)
        trans = cv2.getPerspectiveTransform(contour.bbox.astype("f"), square)
        return cv2.warpPerspective(image, trans, (side, side))
    
    def _square_contour_to_coords(self, contour):
        bl = [a*(contour.side/9.0) for a in range(9)]
        bl_grid = [(x, y) for y in bl for x in bl]
        width = bl[1]
        side = width.astype("i")
        square = np.array(self._square_coordinates((0, 0), side), np.float32)
        grid_elements = [np.array(self._square_coordinates(e, width), np.float32) for e in bl_grid]
        return grid_elements, square, side
        
    def _generate_grid_elements(self, contour):
        """Generate 4 co-ordinates for each element """
        grid_coordinates, _, _ = self._square_contour_to_coords(contour)
       
        digits = '123456789'
        rows = 'ABCDEFGHI'
        cols = digits
        grid_refs = self._cross(rows, cols)
        return dict(zip(grid_refs, grid_coordinates))
    
    def _thresh_img(self, img):
        img_noshading = self._remove_image_shading(img)
        _, thresh = cv2.threshold(img_noshading.astype("uint8"), 128, 255, 
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return thresh

    def _MNIST_preprocess(self, img):
        # get ratio to scale image down to 20x20
        ratio = min(20./img.shape[0], 20./img.shape[1])
        scaleshape = (img.shape[0] * ratio, img.shape[1] * ratio)
        norm = mh.resize.resize_to(img, scaleshape)
    
        # position center of mass of image in a 28x28 field
        dest = np.zeros((28, 28))
        COM = mh.center_of_mass(norm)
        (x, y) = (13.5, 13.5) - COM
        (x, y) = (int(round(x)), int(round(y)))
        dest[x:x + norm.shape[0], y:y + norm.shape[1]] = norm
        return dest
