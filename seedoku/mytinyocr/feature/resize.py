'''
Created on May 30, 2014

@author: minjoon
'''
from .feature_interface import FeatureInterface
import cv2
import numpy as np

class Resize(FeatureInterface):
    '''
    Resize any image to n-by-m image and then flatten the image array.
    self.dim will equal to n*m
    '''

    def __init__(self, xlen=16, ylen=16):
        self.xlen = xlen
        self.ylen = ylen
        self.dim = xlen*ylen
        
    def get_features(self, img):
        '''
        Convert img to xlen-by-ylen array and return flattened array.
        Note that ratio will be conserved.
        '''
        if len(np.shape(img)) == 3:
            img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
        
        ylen, xlen = np.shape(img)
        if ylen > xlen:
            new_ylen = self.ylen
            new_xlen = int(round(float(xlen*self.ylen)/ylen))
        else:
            new_xlen = self.xlen
            new_ylen = int(round(float(ylen*self.xlen)/xlen))
        
        target_img = np.ones((self.xlen,self.ylen))*255
        resized_img = cv2.resize(img, (new_xlen,new_ylen))
        target_img[:new_ylen,:new_xlen] = resized_img
        return target_img.flatten()
