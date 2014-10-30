'''
Created on Jun 1, 2014

@author: minjoon
'''
import os

import cv2

# modules within tinyocr
from .const import ocr_manager_const as omc
from .database import Database

class OCRManager:
    '''
    Connects to database,
    trains, tests, and cross-validates data set.
    Will only handle 
    In order to have several different algorithms of features and learner,
    Instantiate multiple OCRManager objects.
    '''
    def __init__(self, db_path):
        '''
        '''
        self.db = Database(db_path)
    
    def add(self, img, char):
        '''
        Save a new (image,char) pair to the database
        If img is a string, consider it as a filepath to the image.
        Otherwise, it is an array representing the image
        '''
        if isinstance(img, str):
            img = cv2.imread(img)
        self.db.add_img(img,char)
        self.db.commit()
        
    def add_many(self, img_list, char_list):
        '''
        Save a new (image,char) pair to the database
        If img is a string, consider it as a filepath to the image.
        Otherwise, it is an array representing the image
        '''
        self.db.add_img_list(img_list,char_list)
        self.db.commit()
        
    def compute_features(self, feature_alg=None, feature_name=None):
        '''
        Compute features in the database.
        If feature_alg is not specified, use deafult.
        If feature_name is not specified, use feature_alg's class' __name__
        '''
        if feature_alg == None:
            feature_alg = omc.DEFAULT_FEATURE
        self.feature_alg = feature_alg
        if feature_name == None:
            feature_name = type(feature_alg).__name__
        self.feature_name = feature_name # Most recent feature

        imgs = self.db.get_img()
        
        for img_name in imgs:
            img = imgs[img_name]
            features = feature_alg.get_features(img)
            self.db.add_features(feature_name, img_name, nums2str(features))
        self.db.commit()
    
    def set_defaults(self, feature_alg=None, feature_name=None, \
                      learner_alg=None, learner_name=None):
        '''
        If feature name is not given, use most recent by compute_features.
        If learner_name is not given, use the default name.
        If learner_alg is not given, use the default.
        '''

        if feature_alg == None:
            feature_alg = omc.DEFAULT_FEATURE
        self.feature_alg = feature_alg
        if feature_name == None:
            feature_name = type(feature_alg).__name__
        self.feature_name = feature_name # Most recent feature
        if learner_alg == None:
            learner_alg = omc.DEFAULT_LEARNER
        self.learner_alg = learner_alg # Save as most recent learner alg
        if learner_name == None:
            learner_name = learner_alg.__class__.__name__
        self.learner_name = learner_name # Save as most recent learner name
        
    def train_learner(self, feature_name=None, learner_alg=None, \
                      learner_name=None, include=None, exclude=None):
        '''
        Feed data to the learner, fit, and save the model.
        Note that include and exclude cannot be both non-None.
        If feature name is not given, use most recent by compute_features.
        If learner_name is not given, use the default name.
        If learner_alg is not given, use the default.
        '''
        # if both are not None, raise exception
        assert include == None or exclude == None

        if feature_name == None:
            feature_name = self.feature_name
        if learner_alg == None:
            learner_alg = omc.DEFAULT_LEARNER
        self.learner_alg = learner_alg # Save as most recent learner alg
        if learner_name == None:
            learner_name = learner_alg.__class__.__name__
        self.learner_name = learner_name # Save as most recent learner name
    
        imgs = self.db.get_img()
        
        for img_name in imgs:
            if include != None:
                if img_name not in include:
                    continue
            if exclude != None:
                if img_name in exclude:
                    continue
            char = self.db.get_char(img_name).values()[0]
            features = str2nums(self.db.get_features(feature_name, img_name).values()[0])
            learner_alg.add_data(features, char)
            
        learner_alg.fit()
        self.db.add_model(learner_name, learner_alg.__class__.__name__, learner_alg)
        self.db.commit()
    
    def test(self, img, feature_name=None, learner_name=None):
        '''
        Return an array of chars corresponding to the input imgs
        '''
        if isinstance(img, str):
            img = cv2.imread(img)

        if feature_name == None:
            feature_alg = self.feature_alg
        else:
            pass

        if learner_name == None:
            learner_alg = self.learner_alg # Most recent learner
        else:
            learner_alg = self.db.get_model(learner_name).values()[0]
        
        features = feature_alg.get_features(img)
        out = learner_alg.predict(features)
        return out
    
    def loocv(self, feature_name=None, learner_alg=None):
        '''
        Leave-one-out cross-validation.
        Need features to be computed prior to calling this method.
        Given x_1,...,x_n in database,
        Train on x_1,...,x_n-1 and test x_n, so on.
        Skip unique characters. These will be printed out.
        return the percentage of correct answers.
        LOOCV does not save any model on the database at the end.
        '''
        char_counts = self.db.get_char_counts()
        result_chars = {}
        imgs = self.db_get_img()
        _learner_name = self.learner_name
        for img_name in imgs:
            '''
            img_name is the instance that will be left-out
            '''
            char = self.db.get_char(img_name) 
            if char_counts[char] == 1:
                print "%s only occurs once; skipping in LOOCV." %char
            else:
                self.train_learner(feature_name=feature_name, learner_alg=learner_alg, \
                                   learner_name="loocv_temp", exclude=[img_name])
                result_chars[img_name] = self.test(imgs[img_name], feature_name=feature_name, \
                                                   learner_alg=learner_alg)[0]
        
        # Brings back the orignal learner name 
        self.learner_name = _learner_name

        wrong_num = sum([self.db.get_char(img_name)==result_chars[img_name] \
                         for img_name in result_chars])
        return 1-float(wrong_num)/len(self.result_chars)
    
def nums2str(list_input):
    return ','.join(['%f' %x for x in list_input])

def str2nums(str_input):
    return [float(x) for x in str_input.split(',')]
    
if __name__ == "__main__":
    root = "temp"
    if not os.path.exists(root):
        os.mkdir(root)
    db_name = "db.sqlite3"
    db_path = os.path.join(root,db_name)
    om = OCRManager(db_path) 
    om.add(os.path.join(root,'raw/000000.png'),'C')
    om.add(os.path.join(root,'raw/000001.png'),'2')
    om.add(os.path.join(root,'raw/000002.png'),'E')
    om.add(os.path.join(root,'raw/000005.png'),'5')
    om.add(os.path.join(root,'raw/000003.png'),'B')
    om.add(os.path.join(root,'raw/000004.png'),'D')
    om.add(os.path.join(root,'raw/000013.png'),'D')
    om.add(os.path.join(root,'raw/000014.png'),'C')
    om.add(os.path.join(root,'raw/000011.png'),'F')
    om.add(os.path.join(root,'raw/000008.png'),'A')

    om.compute_features()
    om.train_learner()
    om.db.display()
    print om.test(os.path.join(root,'raw/000010.png'))
    om.db.clear()
    