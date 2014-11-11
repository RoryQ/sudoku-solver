'''
Created on Jun 1, 2014

@author: minjoon
'''

# Third-party packages
from sklearn import svm 
import numpy as np

# Within tinyocr
from .learner_interface import LearnerInterface

class SVM(LearnerInterface):
    '''
    SVM interface for OCR
    '''
    
    def __init__(self, **kwargs):
        self.clf = svm.LinearSVC(**kwargs)
        self.X = []
        self.y = []
        self._update_rq = False
    
    def add_data(self, x, y):
        if len(np.shape(x)) == 1:
            self.X.append(x)
            self.y.append(y)
        else:
            self.X.extend(x)
            self.y.extend(y)
        self._update_rq = True
        
    def fit(self):
        self.clf.fit(self.X,self.y)
        
    def predict(self, X):
        '''
        accepts X as a single data point, or array of data points
        If single data point, encompasses it with bracket
        '''
        if len(X.shape) == 1:
            X = [X]
        if self._update_rq:
            self.fit() 
            self._update_rq = False
        return self.clf.predict(X)
        
    def get_scores(self, X):
        if len(X.shape) == 1:
            X = [X]
        if self._update_rq:
            self.fit()
            self._update_rq = False
        return self.clf.decision_function(X)
