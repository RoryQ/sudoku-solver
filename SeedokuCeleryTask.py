import io
from boto.s3.connection import S3Connection
from seedoku import Seedoku
import cv2
import numpy as np
from urllib2 import urlopen
import exifread
from util import fast_unpickle_gzip


class SeedokuTask():
    ocr = None
    seedoku = None
    config = {}

    def __init__(self, config, ocr_path):
        self.ocr = fast_unpickle_gzip(ocr_path)
        self.ocr._update_rq = False
        print 'ocr loaded'

        self.seedoku = Seedoku(self.ocr)
        print 'seedoku created'

        keys = (key for key in config if key.startswith('AWS_SEEDOKU') is True)
        for key in keys:
            self.config[key] = config[key]
        print 'config loaded'

    def numpy_image_from_stringio(self, img_stream, cv2_img_flag=0):
        img_stream.seek(0)

        orientation = self.get_image_orientation(img_stream)

        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2_img_flag)

        return self.rotate_image(image, orientation)

    def generate_temp_s3_url_from_key(self, key):
        s3conn = S3Connection(self.config['AWS_SEEDOKU_WRITE_KEY'],
                              self.config['AWS_SEEDOKU_WRITE_SECRET'])
        bucketname = self.config['AWS_SEEDOKU_S3_BUCKET']
        url = s3conn.generate_url(300, 'GET', bucketname, key)
        print url
        return url

    def get_image_orientation(self, imbytes):
        # get orientation and rewind
        tags = exifread.process_file(imbytes, stop_tag='Orientation')
        imbytes.seek(0)
        return tags.get('Image Orientation')

    def rotate_image(self, image, orientation=None):
        """
        apply rotation based on orientation tag
        """
        # apply orientation if any
        if orientation is not None:
            print orientation
            rot = {'Horizontal (normal)': 0, 'Rotated 180': 2,
                   'Rotated 90 CCW': 3, 'Rotated 90 CW': 1}
            if orientation.printable in rot:
                return np.rot90(image, rot[orientation.printable])
            elif orientation.printable == 'Mirrored horizontal':
                return np.fliplr(image)
            elif orientation.printable == 'Mirrored vertical':
                return np.flipud(image)
            elif orientation.printable == 'Mirrored horizontal then rotated 90 CCW':
                return np.rot90(np.fliplr(image), 1)
            elif orientation.printable == 'Mirrored horizontal then rotated 90 CW':
                return np.rot90(np.fliplr(image), 3)
        return image

    def numpy_image_from_url(self, url, cv2_img_flag=0):
        request = urlopen(url)
        imbytes = io.BytesIO(request.read())

        orientation = self.get_image_orientation(imbytes)

        # convert to numpy image array
        img_array = np.asarray(bytearray(imbytes.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2_img_flag)

        # return rotated image
        return self.rotate_image(image, orientation)

    def aws_upload_key_to_puzzle(self, key):
        img_url = self.generate_temp_s3_url_from_key(key)
        img = self.numpy_image_from_url(img_url, cv2.IMREAD_GRAYSCALE)
        ocrd_puzzle = self.seedoku.image_to_puzzle(img)
        return img, ocrd_puzzle
