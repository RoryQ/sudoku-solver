from celery import Celery
from util import fast_unpickle_gzip
from seedoku import Seedoku


class MyTask(Celery.Task):

    ocr = None
    seedoku = None

    def __init__(self, ocr_path):
        self.ocr = fast_unpickle_gzip(ocr_path)
        self.ocr._update_rq = False
        print 'ocr loaded'

        self.seedoku = Seedoku(self.ocr)
        print 'seedoku created'

    def run(self, img):
        return self.seedoku.image_to_puzzle(img)
