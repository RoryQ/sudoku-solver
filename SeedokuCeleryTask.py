import io
from boto.s3.connection import S3Connection
from seedoku import Seedoku
import cv2
import numpy as np
from urllib2 import urlopen
from util import fast_unpickle_gzip, get_image_orientation, rotate_image


class SeedokuTask():
    seedoku = None
    config = {}

    def __init__(self, config):
        keys = (key for key in config if key.startswith('AWS_SEEDOKU') is True)
        for key in keys:
            self.config[key] = config[key]
        print 'config loaded'

        self.seedoku = Seedoku(config['TESSERACT_LANG'], config['TESSERACT_LIBPATH'], config['TESSERACT_TESSDATA'])
        print 'seedoku created'

    def numpy_image_from_stringio(self, img_stream, cv2_img_flag=0):
        img_stream.seek(0)

        orientation = get_image_orientation(img_stream)

        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2_img_flag)

        return rotate_image(image, orientation)

    def generate_temp_s3_url_from_key(self, key):
        s3conn = S3Connection(self.config['AWS_SEEDOKU_WRITE_KEY'],
                              self.config['AWS_SEEDOKU_WRITE_SECRET'])
        bucketname = self.config['AWS_SEEDOKU_S3_BUCKET']
        url = s3conn.generate_url(300, 'GET', bucketname, key)
        print url
        return url

    def numpy_image_from_url(self, url, cv2_img_flag=0):
        request = urlopen(url)
        imbytes = io.BytesIO(request.read())

        orientation = get_image_orientation(imbytes)

        # convert to numpy image array
        img_array = np.asarray(bytearray(imbytes.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2_img_flag)

        # return rotated image
        return rotate_image(image, orientation)

    def aws_upload_key_to_puzzle(self, key):
        img_url = self.generate_temp_s3_url_from_key(key)
        img = self.numpy_image_from_url(img_url, cv2.IMREAD_GRAYSCALE)
        ocrd_puzzle = self.seedoku.image_to_puzzle(img)
        return img_url, ocrd_puzzle
