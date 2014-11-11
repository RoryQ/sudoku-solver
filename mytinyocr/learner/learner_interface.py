'''
Created on Jun 1, 2014

@author: minjoon
'''
from abc import ABCMeta, abstractmethod

class LearnerInterface:
    '''
    abstract class for learning models
    implicitly assumes online learning,
    since offline learning can be modeled by
    online learning
    '''
    __meta__ = ABCMeta
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def add_data(self, x, y): 
        '''
        If x is 2D, then add batch data
        If x is 1D, then add single data point
        '''
        pass
    
    @abstractmethod
    def predict(self, x, **kwargs):
        '''
        Predict the y value given query x
        input: single data point x
        output: y
        '''
        pass
    
    @abstractmethod
    def get_scores(self, x, **kwargs):
        '''
        return a list of score for each match
        higher score indicates better match
        '''
        pass
    
    
