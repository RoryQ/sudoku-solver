'''
Created on Jun 1, 2014

@author: minjoon
'''
IMG_TABLE = 'img'
MODEL_TABLE = 'model'
IMG_KEY = IMG_TABLE+'_id'
CHAR = 'char'
IMG_BLOB = IMG_TABLE+'_blob'
IMG_ATTR = {IMG_KEY:'INTEGER PRIMARY KEY', CHAR:'text', IMG_BLOB:'array'}
MODEL_KEY = MODEL_TABLE+'_name'
MODEL_ALG = MODEL_TABLE+'_alg'
MODEL_ATTR = {MODEL_KEY:'text', MODEL_ALG:'text'}
FEAT_KEY = IMG_KEY
FEAT_VAL = "feature"
FEAT_ATTR = {FEAT_KEY:'text', FEAT_VAL:'text'}
PICKLE_EXT = 'p'
IMG_DIGIT = 6
IMG_EXT = '.png'
