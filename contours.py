#taken from http://answers.opencv.org/question/6043/segmentation-and-contours/
import cv2
import numpy


class Contour(object):
    """Provides a user-friendly object defining a contour in OpenCV"""

    def __init__(self, cnt):
        self.cnt = cnt
        self.size = len(cnt)

    @property
    def area(self):
        """Contour.area - Area bounded by the contour region"""
        return cv2.contourArea(self.cnt)

    @property
    def perimeter(self):
        """Lorem"""
        return cv2.arcLength(self.cnt, True)

    @property
    def side(self):
        """side"""
        return numpy.ceil(self.perimeter / 4).astype("i")

    @property
    def approx(self):
        """Lorem"""
        return rectify(cv2.approxPolyDP(self.cnt, 0.02 * self.perimeter, True))

    @property
    def hull(self):
        """Lorem"""
        return cv2.convexHull(self.cnt)

    @property
    def moments(self):
        """Lorem"""
        return cv2.moments(self.cnt)

    @property
    def bounding_box(self):
        """Lorem"""
        return cv2.boundingRect(self.cnt)

    @property
    def bbox(self):
        x, y, w, h = cv2.boundingRect(self.cnt)
        return numpy.array(rectangular_coordinates((x, y), w, h), dtype=numpy.int32)


    def shrinkbox(self, percent):
        assert 0 <= percent <= 100
        side = self.perimeter / 4
        smaller_area = self.area / percent
        smaller_side = numpy.sqrt(smaller_area)
        w = smaller_side / 2 
        x, y = self.approx[0]
        return numpy.array(rectangular_coordinates((x+(w/2), y+(w/2)), side-w, side-w), dtype=numpy.int32)

    @property
    def centroid(self):
        if self.moments['m00'] != 0.0:
            cx = self.moments['m10'] / self.moments['m00']
            cy = self.moments['m01'] / self.moments['m00']
            return (cx,cy)
        else:
            return "Region has zero area"

    @property
    def ellipse(self):
        return cv2.fitEllipse(self.cnt)

    @property
    def diameter(self):
        """EquivDiameter: diameter of circle with same area as region"""
        return numpy.sqrt(4 * self.moments['m00'] / numpy.pi)

    def draw_stuff(self, img):
        """
        cv2.drawContours(img, [self.cnt], 0, (0,255,0), 4)
        cv2.drawContours(img, [self.approx], 0, (255,0,0), 2)
        cv2.drawContours(img, [self.hull], 0, (0,0,255), 2)
        #cv2.drawContours(img, [self.bbox], 0, (0, 255, 255), 2)
        message = 'green : original contour'
        cv2.putText(img, message, (20,20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0))
        message = 'blue : approximated contours'
        cv2.putText(img, message, (20,40), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0))
        message = 'red : convex hull'
        cv2.putText(img, message,(20,60), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255))
        message = 'yellow : bounding box'
        cv2.putText(img, message,(20,80), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,255))
        """
        cv2.circle(img, (int(self.centroid[0]), int(self.centroid[1])), 1, (0, 0, 0), -1)

def rectangular_coordinates(bottom_left, width, height):
    (x, y), w, h = bottom_left, width, height
    return [x, y], [x, y+h], [x+w, y+h], [x+w, y]  # BL TL TR BR

def rectify(rect):
    """Sorts rectangular contour co-ords. CW from BL"""
    if rect.shape[0] != 4:
        return rect
    ind = numpy.argsort(rect[:, 0][:, 0])
    left, right = rect[ind][:2], rect[ind][2:]
    bl, tl = left[numpy.argsort(left[:, 0][:, 1])]
    br, tr = right[numpy.argsort(right[:, 0][:, 1])]
    return numpy.vstack((bl, tl, tr, br))
