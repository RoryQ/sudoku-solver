import mytinyocr, mahotas as mh, numpy as np, timeit

def process_img(img):
    ratio = min(20./img.shape[0], 20./img.shape[1])
    scaleshape = (img.shape[0] * ratio, img.shape[1] * ratio)
    norm = mh.resize.resize_to(img, scaleshape)
    
    dest = np.zeros((28, 28))
    COM = mh.center_of_mass(norm)
    (x, y) = (13.5, 13.5) - COM
    (x, y) = (int(round(x)), int(round(y)))
    dest[x:x + norm.shape[0], y:y + norm.shape[1]] = norm
    
    return dest

print "loading ocrdb"
start = timeit.default_timer()
ocr = mytinyocr.OCRManager('ocrdb')
print timeit.default_timer() - start

print "loading defaults"
start = timeit.default_timer()
ocr.set_defaults(feature_alg=mytinyocr.feature.resize.Resize(), learner_alg=ocr.db.get_model('SVM').values()[0])
print timeit.default_timer() - start

print "processing single test image"
start = timeit.default_timer()
seven = process_img(mh.imread('digits/7.png'))
print ocr.test(seven)
print timeit.default_timer() - start

