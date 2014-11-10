import cv2
import numpy as np
from contours import Contour
import pymorph as pm
import mahotas as mh
from tinyocr.feature.resize import Resize

class Seedoku(object):
    """
    Sudoku puzzle detection class.
    Uses computer vision techniques to detect and isolate a sudoku puzzle
    grid in an image, which then extracts the puzzle elements to be fed
    into an OCR.

    Seedoku(ocr_alg) -> returns new Seedoku object that uses the supplied
    ocr algorithm to predict puzzle element digit values. Uses Resize
    from tinyocr.feature.resize as the default feature algorithm

    Seedoku(ocr_alg, feature_alg) -> same as above, but allows a custom
    feature detection algorithm
    """

    def __init__(self, ocr_alg, feature_alg=None):
        if feature_alg is None:
            feature_alg = Resize()
        self.feature_alg = feature_alg
        self.ocr_alg = ocr_alg


    def image_to_puzzle(self, img):
        """
        Applys OCR to image and returns a dictionary predicted
        puzzle grid elements.

        Parameters
        ----------
        img : ndarray
            Any integer image type

        Returns
        -------
        dict(key, value) : string, string
            key is the grid_id where grid rows are letters A-I
            and columns are numbers 1-9. e.g.

            A1 A2 A3 | A4 A5 A6 | A7 A8 A9
            B1 B2 B4 | B4 B5 B6 | B7 B8 B9
            C1 ...
        """
        crop_img, crop_contour = self._crop_to_puzzle_area(img)
        grid_elems = self._generate_grid_elements(crop_contour)

        puzzle = {}
        for elem_id, elem_contour in grid_elems.items():
            elem_img = self._get_grid_img(crop_img, elem_contour)

            if reduce(lambda x, y: x*y, elem_img.shape) < 500:
                puzzle[elem_id] = "0"
                continue
            elem_img = pm.binary(elem_img).astype("float32")
            elem_img = self._MNIST_preprocess(elem_img)
            features = self.feature_alg.get_features(elem_img)
            guess = self.ocr_alg.predict(features)
            puzzle[elem_id] = str(guess[0])

        return puzzle

    def _get_grid_img(self, crop_img, elem_contour):
        cnt = Contour(elem_contour)
        halfbox = cnt.shrinkbox(100).astype("f")
        square = self._square_coordinates(cnt.side)
        trans = cv2.getPerspectiveTransform(halfbox, square)
        box = cv2.warpPerspective(crop_img, trans, (cnt.side, cnt.side))
        _, thresh = cv2.threshold(box, 127, 255, 0)
        binary = pm.binary(thresh)
        box = binary.astype(int) * 255
        box = pm.edgeoff(box)
        box = mh.croptobbox(box)
        return box

    def _crop_to_puzzle_area(self, img):
        downsize = self._downsize_image(img)
        thresh = self._thresh_img(downsize)
        contours = self._get_contours(thresh.copy())
        biggest = self._get_biggest_rect(contours)
        warp = self._warp_to_contour(thresh, biggest)
        return warp, biggest

    def _square_coordinates(self, width, bottom_left=None):
        """
        returns ndarray of co-ordinates of a square clockwise from 
        bottom left
        """
        if bottom_left is None:
            bottom_left = (0, 0)
        (x, y), w = bottom_left, width
        coords = [x, y], [x, y+w], [x+w, y+w], [x+w, y]  # BL TL TR BR
        return np.array(coords, np.float32)

    def _cross(self, A, B):
        """cross product of elements in a and elements in b"""
        return [a + b for a in A for b in B]
    
    def _downsize_image(self, img):
        if sum(img.shape[:2]) > 1500:
            small = map((lambda a: int(a /(sum(img.shape[:2]) / 1500.0))), 
                        img.shape[:2])
            return cv2.resize(img, tuple(small))
        return img
    
    def _remove_image_shading(self, image, pixelk=None, ksize=None):
        """
        Improves contrast of image by removing shaing in image.

        Applys morphological closing on image to extract background
        (this removes thin elements e.g. text and lines)
        then divide original image to remove background shading
        """
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
    
    def _warp_to_contour(self, image, contour):
        side = contour.side
        square = self._square_coordinates(side)
        trans = cv2.getPerspectiveTransform(contour.bbox.astype("f"), square)
        return cv2.warpPerspective(image, trans, (side, side))
    
    def _square_contour_to_coords(self, contour):
        bl = [a*(contour.side/9.0) for a in range(9)]
        bl_grid = [(x, y) for y in bl for x in bl]
        width = bl[1]
        side = width.astype("i")
        grid_elements = [self._square_coordinates(side, e) for e in bl_grid]
        return grid_elements, side
        
    def _generate_grid_elements(self, contour):
        """Generate 4 co-ordinates for each element """
        grid_coordinates, side = self._square_contour_to_coords(contour)
       
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
        """
        Applys image processing that mirrors the preprocessing of the
        MNIST dataset.
        Assumes image has been cropped to bounding box.

        Scales image down to 20 x 20 then centered in a 28 x 28 grid
        based on the centre of mass of input image
        """
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
