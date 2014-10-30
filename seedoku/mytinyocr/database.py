'''
Created on Jun 1, 2014

@author: minjoon
'''
# default packages
import os
import cPickle as pickle
import sqlite3
import shutil
import io
from collections import defaultdict

# 3rd-party packages
import cv2
import numpy as np

# modules within tinyocr
from .const import database_const as dbc

class Database:
    '''
    Database manager for tinyocr.
    Has three functionalities:
    1. Add/get character image
    2. Add/get learning model (e.g. SVM)
    3. Add/get features for specified image and model
    '''
    
    def __init__(self, db_path, clear=False):
        '''
        if db_path exists,
        connects to the existing database.
        
        if not:
        Saves images to name_img.
        
        "img" table has two attributes:
        img_name, char
        "model" table has two attributes:
        model_name, model_alg
        
        each model corresponds to model_name table,
        which contains features of the model
        It has two attributes:
        img_name, features (separated by comma)
        
        set clear=True to clean all folders and db
        already present with the given db_path.
        '''
        
        self.db_path = db_path
        curr_dir = os.path.dirname(db_path)
        db_name = os.path.basename(db_path)
        name = '.'.join(db_name.split('.')[:-1])
        img_foldername = name+'_'+dbc.IMG_TABLE;
        model_foldername = name+'_'+dbc.MODEL_TABLE;
        self.img_dir = os.path.join(curr_dir, img_foldername)
        self.model_dir = os.path.join(curr_dir, model_foldername)
        
        if clear:
            self.clear()
        
        #converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, _adapt_array)
        sqlite3.register_converter("array", _convert_array)
        
        #Connect to the db
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        
        
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        
        table_names = self._get_table_names()
        if dbc.IMG_TABLE not in table_names:
            self._create_table(dbc.IMG_TABLE, dbc.IMG_ATTR)
        if dbc.MODEL_TABLE not in table_names:
            self._create_table(dbc.MODEL_TABLE, dbc.MODEL_ATTR)
        
    def add_img(self, img, char):
        '''
        adds image to database as ndarray type
        '''
        c = self.conn.cursor()
        # Get the top image id
        command = "SELECT %s FROM %s ORDER BY %s DESC LIMIT 1" % (dbc.IMG_KEY,dbc.IMG_TABLE,dbc.IMG_KEY)        
        c.execute(command)
        img_id = c.fetchone() or 0
        img_id += 1
        # Add the (name,char) pair to the 'img' table
        command = ("INSERT INTO %s('%s','%s','%s') VALUES('%d','%s', ?)" \
                   %(dbc.IMG_TABLE, dbc.IMG_KEY, dbc.CHAR, dbc.IMG_BLOB, img_id, char))
        c.execute(command, (img,) )
        return img_id
        
    def add_img_list(self, img_list, char_list):
        '''
        same as add_img but processes list of images in a sqlite transaction
        '''
        c = self.conn.cursor()
        img_id_list = []
        # Get the top image id
        command = "SELECT %s FROM %s ORDER BY %s DESC LIMIT 1" % (dbc.IMG_KEY,dbc.IMG_TABLE,dbc.IMG_KEY)        
        c.execute(command)
        img_id = c.fetchone() or 0
        
        for i in xrange(img_list.shape[0]):
            # Add the (name,char) pair to the 'img' table
            
            img_id += 1
            command = ("INSERT INTO %s('%s','%s','%s') VALUES('%d','%s', ?)" \
                       %(dbc.IMG_TABLE, dbc.IMG_KEY, dbc.CHAR, dbc.IMG_BLOB, img_id, char_list[i]))
            c.execute(command, (img_list[i],) )
            img_id_list.append(img_id)
        
        # Returns the imd_ids of the image
        return img_id_list
        
    def get_img(self, img_name=None, char=None, grayscale=True):
        '''
        Fetch all data that satisfies the given condition.
        If no condition is specified, all data are returned
        output type is a dictionary,
        where key is the img_name and the value is the image array.
        '''
        c = self.conn.cursor()
        # Query the img_name in the img table
        command = "SELECT %s, %s FROM %s" %(dbc.IMG_KEY, dbc.IMG_BLOB,dbc.IMG_TABLE)
        conditions = []
        if img_name != None:
            conditions.append("%s='%s'" %(dbc.IMG_KEY,img_name))
        if char != None:
            conditions.append("%s='%s'" %(dbc.CHAR,char))
        if len(conditions) > 0:
            command += " WHERE " + " AND ".join(conditions)
        c.execute(command)
        imgs = dict(c.fetchall())
        
        return imgs
    
    def get_char(self, img_name=None):
        '''
        Return a dictionary where
        key is the img_name, and value is the char.
        If img_name is specified, will return
        single-element dictionary
        '''
        c = self.conn.cursor()
        command = "SELECT %s,%s FROM %s" %(dbc.IMG_KEY,dbc.CHAR,dbc.IMG_TABLE)
        if img_name != None:
            command += " WHERE %s='%s'" %(dbc.IMG_KEY,img_name)
        c.execute(command)
        chars = {elem[0]:elem[1] for elem in c.fetchall()}
        return chars
    
    def add_model(self, model_name, model_alg, model_object):
        '''
        Add a model to the "model" table.
        Also create a pickle object in the model folder.
        If a model with the same name exists, replace it.
        '''
        c = self.conn.cursor()
        self.delete_model(model_name)
        
        # Add (model_name, model_alg) to the "model" table
        c.execute("INSERT INTO %s(%s,%s) VALUES ('%s','%s')" \
                  %(dbc.MODEL_TABLE,dbc.MODEL_KEY,dbc.MODEL_ALG,model_name,model_alg))
        # Save pickle file to the model folder (overwrite)
        self._dump_pickle(model_object, self._get_pickle_path(model_name))

        
    def delete_model(self, model_name):
        '''
        Remove corresponding row from the model table.
        Delete pickle object.
        '''
        # Remove row
        c = self.conn.cursor()
        c.execute("DELETE FROM %s WHERE %s='%s'" %(dbc.MODEL_TABLE,dbc.MODEL_KEY,model_name))
       
        # Remove pickle object
        pickle_path = self._get_pickle_path(model_name) 
        if os.path.exists(pickle_path):
            os.remove(pickle_path)
        
    def get_model(self, model_name=None, model_alg=None, entire=False):
        '''
        If no condition is specified, either all or latest model is returned,
        depending on the value of 'entire'
        '''
        c = self.conn.cursor()
        conditions = []
        command = "SELECT %s FROM %s" %(dbc.MODEL_KEY,dbc.MODEL_TABLE)
        if model_name != None:
            conditions.append("%s='%s'" %(dbc.MODEL_KEY,model_name))
        if model_alg != None:
            conditions.append("%s='%s'" %(dbc.MODEL_ALG,model_alg))
        if len(conditions) > 0:
            command += " WHERE " + " AND ".join(conditions)
            
        c.execute(command)
        model_names = [elem[0] for elem in c.fetchall()]
        models = {}
        if not entire:
            model_names = model_names[-1:]
        for model_name in model_names:
            models[model_name] = self._load_pickle(self._get_pickle_path(model_name))
        return models
        
    def add_features(self, feature_name, img_name, features):
        '''
        Add features of img_name extracted to model_name table.
        Note that features should be a string. Conversion to other formats
        is up to the user's discretion.
        If same img_name already exists, replace it.
        '''
        c = self.conn.cursor()
        if feature_name not in self._get_table_names():
            self._create_table(feature_name,dbc.FEAT_ATTR)
        
        # Delete existing row with img_name
        command = "DELETE FROM %s WHERE %s='%s'" %(feature_name,dbc.FEAT_KEY,img_name)
        c.execute(command)
        
        # Add new features
        c.execute("INSERT INTO %s VALUES('%s','%s')" %(feature_name,img_name,features))
        
    def get_features(self, feature_name, img_name=None):
        '''
        Returns a dictionary where key is the img name, val is the feature
        '''
        c = self.conn.cursor()
        command = "SELECT %s,%s FROM %s" %(dbc.FEAT_KEY,dbc.FEAT_VAL,feature_name)
        if img_name != None:
            command += " WHERE %s='%s'" %(dbc.FEAT_KEY,img_name)
        c.execute(command)
        features_list = {elem[0]:elem[1] for elem in c.fetchall()}
        assert len(features_list) < 2 # remove this line when functionality is added.
        return features_list
    
    def delete_features(self, feature_name):
        c = self.conn.cursor()
        if feature_name in self._get_table_names():
            command = "DROP %s" %(feature_name)
            c.execute(command)
    
    def get_char_counts(self):
        chars = self.get_char()
        char_counts = defaultdict(int)
        for char in chars:
            char_counts[char] += 1
        return char_counts
        
    def clear(self):
        '''
        Clear img and model folder and database
        '''
        if os.path.exists(self.img_dir):
            shutil.rmtree(self.img_dir)
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def close(self):
        '''
        Commit the current change and close the connection.
        '''
        self.conn.commit()
        self.conn.close()
        
    def commit(self):
        self.conn.commit()
        
    def display(self):
        '''
        Print list of tables
        '''
        c = self.conn.cursor()
        for table_name in self._get_table_names():
            print(table_name)
            c.execute("SELECT * FROM %s" %table_name)
            for x in c.fetchall():
                print x
            print("")
             
        
    '''
    These functions are for internal use only.
    '''
    def _get_table_names(self):
        '''
        Obtain all table names in the database
        '''
        c = self.conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        # convert 2D array to 1D array
        table_names = [elem[0] for elem in c.fetchall()]
        return table_names
    
    def _create_table(self, table_name, attr_dict):
        c = self.conn.cursor()
        attr_str = ','.join(["%s %s" %(key,attr_dict[key]) for key in attr_dict])
        command = 'CREATE TABLE %s(%s)' %(table_name,attr_str)
        c.execute(command)
        
    def _get_pickle_path(self, model_name):
        return os.path.join(self.model_dir,"%s.%s" %(model_name,dbc.PICKLE_EXT))
    
    def _dump_pickle(self, p, filepath):
        f = open(filepath, 'wb')
        pickle.dump(p, f)
        f.close()
        
    def _load_pickle(self, filepath):
        f = open(filepath, 'rb')
        p = pickle.load(f)
        f.close()
        return p
        
def _adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    # http://stackoverflow.com/a/3425465/190597 (R. Hill)
    return buffer(out.read())

def _convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
            
def next_name(path, digit, prefix='', suffix='', start_num=0):
    '''
    To be moved to a separate util folder.
    '''
    dir_list = os.listdir(path)
    start_num = 0
    name = prefix + ('{0:0%d}'%digit).format(start_num) + suffix
    while name in dir_list:
        start_num += 1
        name = prefix + ('{0:0%d}'%digit).format(start_num) + suffix
    return name

if __name__ == "__main__":
    db = Database('test.db',True)
    
    model_name = 'myModel'
    model_alg = 'tuple'
    img = np.array([[1,0],[0,1]])
    model = (1,2,3)
    features='3,4,5'
    feature_name = 'naive'
    char = 'h'
    
    # Test add/get image functionality
    img_name = db.add_img(img,char)
    new_imgs = db.get_img(img_name=img_name,char=char)
    if len(new_imgs) > 0:
        print list(img.flatten()) == list(new_imgs.values()[0].flatten())
    else:
        print False    
        
    # Test get char functionality
    new_chars = db.get_char(img_name)
    if len(new_chars) > 0:
        print char == new_chars.values()[0]
    else:
        print False
    
    # Test add/get model functionality
    db.add_model(model_name,model_alg,model)
    new_models = db.get_model(model_name=model_name)
    if len(new_models) > 0:
        print model == new_models.values()[0]
    else:
        print False
    
    # Test add/get feature functionality
    db.add_features(feature_name,img_name,features)
    new_features = db.get_features(feature_name,img_name)
    if len(new_features) > 0:
        print features == new_features.values()[0]
    else:
        print False
        
    db.display()
    
    db.close() 
    