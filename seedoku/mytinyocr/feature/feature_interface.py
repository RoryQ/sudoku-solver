'''
Created on May 30, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod


class FeatureInterface:
    '''
    interface for feature extraction methods.
    init method sets up parameters.
    
    img -> n-dimensional vector
    '''
    
    __metaclass__ = ABCMeta
    
    def __init__(self, **kwargs):
        '''
        defines parameters
        '''
        self.dim = 0 # feature dimension
            
    @abstractmethod
    def get_features(self, img, **kwargs):
        '''
        input: img in 2D numpy array
        output: self.dim-dimension array consisting of features
        '''
        pass
    
    